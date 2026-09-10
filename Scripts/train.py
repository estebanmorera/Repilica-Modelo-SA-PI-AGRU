# -*- coding: utf-8 -*-
"""Entrena, evalua y documenta la corrida C1/B0005/seed 58.

Uso normal::

    python preprocessing.py
    python train.py

La metrica principal de este repositorio se calcula con un voto por ciclo.  El
R2 de curva completa se conserva como diagnostico comparable con el reporte
historico; el ultimo 15 % se informa aparte para no ocultar la extrapolacion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

# La corrida historica uso esta configuracion de determinismo para CUDA/cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn

from config import CONFIG
from model import SA_PI_AGRU


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    """Fija las fuentes de azar usadas por Python, NumPy, CPU y CUDA."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_battery(battery_id: str) -> dict:
    """Carga un NPZ procesado y devuelve sus arreglos y metadatos."""

    path = CONFIG.processed_dir / f"{battery_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}; ejecute primero preprocessing.py")
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files if key != "metadata_json"}
        result["metadata"] = json.loads(str(data["metadata_json"]))
    return result


def tensor_split(data: dict, split: str) -> dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(np.asarray(data[f"{key}_{split}"])).float()
        for key in ("X", "t", "u", "cycle", "weight")
    }


def one_sequence_per_cycle(
    cycles: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Escoge una ventana al azar por ciclo; cada ciclo aporta un voto."""

    flat_cycles = cycles.flatten().to(torch.int64)
    selected = []
    for cycle_id in torch.unique(flat_cycles, sorted=True):
        candidates = torch.nonzero(flat_cycles == cycle_id, as_tuple=False).flatten()
        choice = torch.randint(len(candidates), (1,), generator=generator)
        selected.append(candidates[choice])
    return torch.cat(selected)


def make_collocation_pool(
    train_data: dict[str, torch.Tensor], generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crea los 5 000 pares X/t independientes usados por la perdida fisica."""

    count = CONFIG.physics.collocation_points
    cycles = train_data["cycle"].flatten().to(torch.int64)
    unique_cycles = torch.unique(cycles, sorted=True)
    chosen = []
    for position in range(count):
        cycle_id = unique_cycles[position % len(unique_cycles)]
        candidates = torch.nonzero(cycles == cycle_id, as_tuple=False).flatten()
        choice = torch.randint(len(candidates), (1,), generator=generator)
        chosen.append(candidates[choice])
    indices = torch.cat(chosen)
    order = torch.randperm(count, generator=generator)
    sensors = train_data["X"][indices[order]].contiguous()
    times = (
        torch.rand((count, 1), generator=generator)
        * CONFIG.physics.collocation_time_max
    )
    return sensors, times


def physics_loss(
    model: SA_PI_AGRU,
    sensors: torch.Tensor,
    times: torch.Tensor,
    initial_sensors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Residuo de Verhulst mas la condicion inicial del paper."""

    times = times.detach().clone().requires_grad_(True)
    predicted_u = model(sensors, times)
    derivative = torch.autograd.grad(
        predicted_u,
        times,
        grad_outputs=torch.ones_like(predicted_u),
        create_graph=True,
        retain_graph=True,
    )[0]

    physics = CONFIG.physics
    shifted_u = predicted_u - physics.offset
    right_side = model.r * shifted_u * (
        1.0 - shifted_u / (physics.carrying_capacity - physics.offset)
    )
    residual_mse = torch.mean((derivative - right_side) ** 2)

    zero_time = torch.zeros_like(times[:1])
    initial_u = model(initial_sensors[:1], zero_time)
    initial_mse = torch.mean((initial_u - physics.initial_loss) ** 2)
    total = residual_mse + physics.initial_condition_weight * initial_mse
    return total, residual_mse, initial_mse


@torch.no_grad()
def predict_windows(
    model: SA_PI_AGRU, arrays: dict[str, torch.Tensor]
) -> np.ndarray:
    """Predice SoH por ventana, sin dropout y en lotes manejables."""

    model.eval()
    predictions = []
    batch_size = CONFIG.train.batch_size
    for start in range(0, len(arrays["X"]), batch_size):
        sensors = arrays["X"][start : start + batch_size].to(DEVICE)
        times = arrays["t"][start : start + batch_size].to(DEVICE)
        predicted_soh = 1.0 - model(sensors, times)
        if not torch.isfinite(predicted_soh).all():
            raise FloatingPointError("La prediccion contiene NaN o infinito")
        predictions.append(predicted_soh.cpu().numpy())
    return np.concatenate(predictions).reshape(-1)


def aggregate_by_cycle(
    arrays: dict[str, torch.Tensor], window_predictions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Promedia las ventanas para obtener un unico valor por ciclo."""

    cycle_ids = arrays["cycle"].numpy().reshape(-1).astype(int)
    actual_soh = 1.0 - arrays["u"].numpy().reshape(-1)
    rows = []
    for cycle_id in np.unique(cycle_ids):
        mask = cycle_ids == cycle_id
        rows.append(
            (
                cycle_id,
                float(np.mean(actual_soh[mask])),
                float(np.mean(window_predictions[mask])),
            )
        )
    matrix = np.asarray(rows, dtype=np.float64)
    return matrix[:, 0].astype(int), matrix[:, 1], matrix[:, 2]


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    """RMSE, MAE y R2 sin recortar valores negativos."""

    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if actual.shape != prediction.shape or not actual.size:
        raise ValueError("Los vectores real y predicho deben coincidir")
    error = actual - prediction
    sse = float(np.sum(error**2))
    mean = float(np.mean(actual))
    sst = float(np.sum((actual - mean) ** 2))
    r2 = None if sst <= np.finfo(np.float64).eps else 1.0 - sse / sst
    if r2 is not None and r2 > 1.0 + 1e-10:
        raise AssertionError("R2 imposible; revise el emparejamiento de datos")
    return {
        "n_cycles": int(len(actual)),
        "rmse_soh": float(np.sqrt(np.mean(error**2))),
        "rmse_percent_points": float(100.0 * np.sqrt(np.mean(error**2))),
        "mae_soh": float(np.mean(np.abs(error))),
        "mae_percent_points": float(100.0 * np.mean(np.abs(error))),
        "r2": None if r2 is None else float(r2),
        "r2_percent": None if r2 is None else float(100.0 * r2),
        "sse": sse,
        "sst": sst,
    }


def evaluate_split(
    model: SA_PI_AGRU, arrays: dict[str, torch.Tensor], split_name: str
) -> tuple[dict, list[dict]]:
    predictions = predict_windows(model, arrays)
    cycles, actual, predicted = aggregate_by_cycle(arrays, predictions)
    rows = [
        {
            "cycle": int(cycle),
            "soh_true": float(truth),
            "soh_pred": float(estimate),
            "error_soh": float(estimate - truth),
            "split": split_name,
        }
        for cycle, truth, estimate in zip(cycles, actual, predicted)
    ]
    return regression_metrics(actual, predicted), rows


def validation_cycle_mse(
    model: SA_PI_AGRU, validation_data: dict[str, torch.Tensor]
) -> float:
    predictions = predict_windows(model, validation_data)
    _, actual, predicted = aggregate_by_cycle(validation_data, predictions)
    return float(np.mean((actual - predicted) ** 2))


def checkpoint_payload(
    model: SA_PI_AGRU,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_validation_mse: float,
    history: list[dict],
    generator: torch.Generator,
    collocation_cursor: int,
) -> dict:
    payload = {
        "run_name": CONFIG.run_name,
        "seed": CONFIG.train.seed,
        "epoch": epoch,
        "optimizer_updates": epoch,
        "best_validation_cycle_mse": best_validation_mse,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "generator_state": generator.get_state(),
        "collocation_cursor": collocation_cursor,
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    return payload


def save_checkpoint(payload: dict, destination: Path) -> None:
    """Guarda de forma atomica para no dejar un checkpoint a medias."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:  # PyTorch 1.13.1 no conoce weights_only.
        return torch.load(path, map_location=DEVICE)


def train_model(resume: bool = False) -> tuple[SA_PI_AGRU, list[dict]]:
    """Ejecuta las 10 000 actualizaciones deterministas de C1."""

    seed_everything(CONFIG.train.seed)
    battery = load_battery(CONFIG.data.train_battery)
    train_data = tensor_split(battery, "train")
    validation_data = tensor_split(battery, "val")

    model = SA_PI_AGRU().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.train.learning_rate,
        weight_decay=CONFIG.train.weight_decay,
    )
    generator = torch.Generator().manual_seed(CONFIG.train.seed + 9_107)
    collocation_sensors, collocation_times = make_collocation_pool(
        train_data, generator
    )

    flat_cycles = train_data["cycle"].flatten()
    first_cycle_indices = torch.nonzero(
        flat_cycles == torch.min(flat_cycles), as_tuple=False
    ).flatten()
    initial_index = int(first_cycle_indices[len(first_cycle_indices) // 2].item())
    initial_sensors = train_data["X"][initial_index : initial_index + 1]

    last_path = CONFIG.output_dir / "checkpoint_last.pt"
    start_epoch = 0
    best_validation_mse = math.inf
    history: list[dict] = []
    collocation_cursor = 0
    if last_path.exists():
        if not resume:
            raise FileExistsError(
                f"Ya existe {last_path}. Use --resume o retire esa salida."
            )
        state = load_checkpoint(last_path)
        if state.get("run_name") != CONFIG.run_name or state.get("seed") != 58:
            raise ValueError("El checkpoint no pertenece a C1/B0005/seed58")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"])
        best_validation_mse = float(state["best_validation_cycle_mse"])
        history = list(state.get("history", []))
        collocation_cursor = int(state.get("collocation_cursor", 0))
        generator.set_state(state["generator_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])

    print(
        f"run={CONFIG.run_name} seed=58 device={DEVICE} "
        f"train_cycles={len(torch.unique(train_data['cycle']))}"
    )
    for epoch_index in range(start_epoch, CONFIG.train.epochs):
        model.train()
        indices = one_sequence_per_cycle(train_data["cycle"], generator)
        indices = indices[torch.randperm(len(indices), generator=generator)]
        totals = {"data": 0.0, "physics": 0.0, "residual": 0.0, "initial": 0.0}
        batches = 0

        for start in range(0, len(indices), CONFIG.train.batch_size):
            batch_indices = indices[start : start + CONFIG.train.batch_size]
            sensors = train_data["X"][batch_indices].to(DEVICE)
            times = train_data["t"][batch_indices].to(DEVICE)
            targets = train_data["u"][batch_indices].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(sensors, times)
            data_mse = torch.mean((prediction - targets) ** 2)

            count = len(batch_indices)
            positions = (
                torch.arange(collocation_cursor, collocation_cursor + count)
                % CONFIG.physics.collocation_points
            )
            collocation_cursor = int(
                (collocation_cursor + count) % CONFIG.physics.collocation_points
            )
            physical_mse, residual_mse, initial_mse = physics_loss(
                model,
                collocation_sensors[positions].to(DEVICE),
                collocation_times[positions].to(DEVICE),
                initial_sensors.to(DEVICE),
            )
            total_loss = data_mse + CONFIG.physics.residual_weight * physical_mse
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Loss invalida en epoch {epoch_index + 1}")

            total_loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                CONFIG.train.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            totals["data"] += float(data_mse.detach())
            totals["physics"] += float(physical_mse.detach())
            totals["residual"] += float(residual_mse.detach())
            totals["initial"] += float(initial_mse.detach())
            batches += 1

        epoch = epoch_index + 1
        validation_mse = None
        new_best = False
        if epoch % CONFIG.train.validation_every == 0 or epoch == CONFIG.train.epochs:
            validation_mse = validation_cycle_mse(model, validation_data)
            if validation_mse < best_validation_mse:
                best_validation_mse = validation_mse
                new_best = True

        record = {
            "epoch": epoch,
            "optimizer_updates": epoch,
            "data_mse": totals["data"] / batches,
            "physics_loss": totals["physics"] / batches,
            "residual_mse": totals["residual"] / batches,
            "initial_mse": totals["initial"] / batches,
            "val_cycle_mse": validation_mse,
            "r_normalized": float(model.r.detach()),
        }
        history.append(record)
        payload = checkpoint_payload(
            model,
            optimizer,
            epoch,
            best_validation_mse,
            history,
            generator,
            collocation_cursor,
        )
        if new_best:
            save_checkpoint(payload, CONFIG.output_dir / "checkpoint_best.pt")
        if epoch % CONFIG.train.checkpoint_every == 0 or epoch == CONFIG.train.epochs:
            save_checkpoint(payload, last_path)
        if epoch == 1 or epoch % CONFIG.train.log_every == 0:
            print(json.dumps(record, sort_keys=True))

    save_checkpoint(
        checkpoint_payload(
            model,
            optimizer,
            CONFIG.train.epochs,
            best_validation_mse,
            history,
            generator,
            collocation_cursor,
        ),
        CONFIG.output_dir / "checkpoint_endpoint.pt",
    )
    return model, history


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_cycles(path: Path, battery_id: str, rows: list[dict]) -> None:
    """Guarda una figura sencilla de SoH real frente a predicho."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cycles = [row["cycle"] for row in rows]
    actual = [row["soh_true"] for row in rows]
    predicted = [row["soh_pred"] for row in rows]
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(cycles, actual, "k--", linewidth=1.5, label="SoH real")
    axis.plot(cycles, predicted, color="#d62728", linewidth=1.5, label="SoH predicho")
    axis.set(title=f"SA-PI-AGRU: {battery_id}", xlabel="Ciclo", ylabel="SoH")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def evaluate_model(model: SA_PI_AGRU) -> dict:
    """Evalua B0005 y la transferencia zero-shot a B0006/B0007."""

    destination = CONFIG.output_dir / "evaluation_endpoint"
    destination.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[str, dict] = {}

    train_battery = load_battery(CONFIG.data.train_battery)
    full_rows: list[dict] = []
    for split in ("train", "val", "test"):
        metrics, rows = evaluate_split(model, tensor_split(train_battery, split), split)
        key = f"B0005_{split}"
        all_metrics[key] = metrics
        full_rows.extend(rows)
        write_csv(destination / f"{key}_cycles.csv", rows)
    full_rows.sort(key=lambda row: row["cycle"])
    full_actual = np.asarray([row["soh_true"] for row in full_rows])
    full_prediction = np.asarray([row["soh_pred"] for row in full_rows])
    all_metrics["B0005_full_DIAGNOSTIC_ONLY"] = regression_metrics(
        full_actual, full_prediction
    )
    write_csv(destination / "B0005_full_cycles.csv", full_rows)
    plot_cycles(destination / "B0005.png", "B0005", full_rows)

    for battery_id in CONFIG.data.test_batteries:
        arrays = tensor_split(load_battery(battery_id), "test")
        metrics, rows = evaluate_split(model, arrays, "external")
        all_metrics[battery_id] = metrics
        write_csv(destination / f"{battery_id}_cycles.csv", rows)
        plot_cycles(destination / f"{battery_id}.png", battery_id, rows)

        count = len(rows)
        train_end = min(max(1, int(count * 0.70)), count - 2)
        tail_start = min(max(train_end + 1, int(count * 0.85)), count - 1)
        tail_rows = rows[tail_start:]
        tail_key = f"{battery_id}_tail15_PRIMARY"
        tail_actual = np.asarray([row["soh_true"] for row in tail_rows])
        tail_prediction = np.asarray([row["soh_pred"] for row in tail_rows])
        all_metrics[tail_key] = regression_metrics(tail_actual, tail_prediction)
        write_csv(destination / f"{tail_key}_cycles.csv", tail_rows)

    result = {
        "run_name": CONFIG.run_name,
        "seed": CONFIG.train.seed,
        "checkpoint": "endpoint",
        "checkpoint_epoch": CONFIG.train.epochs,
        "metrics": all_metrics,
    }
    (destination / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def metrics_from_csv(path: Path) -> dict:
    """Recalcula las metricas de un CSV historico sin confiar en su JSON."""

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    actual = np.asarray([float(row["soh_true"]) for row in rows])
    predicted = np.asarray([float(row["soh_pred"]) for row in rows])
    return regression_metrics(actual, predicted)


def verify_reference() -> dict:
    """Comprueba que CSV y metrics.json del resultado congelado concuerdan."""

    metrics_path = CONFIG.reference_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Falta la referencia: {metrics_path}")
    stored = json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]
    files = {
        "B0005_full_DIAGNOSTIC_ONLY": "B0005_full_cycles.csv",
        "B0005_test": "B0005_test_cycles.csv",
        "B0006": "B0006_cycles.csv",
        "B0006_tail15_PRIMARY": "B0006_tail15_PRIMARY_cycles.csv",
        "B0007": "B0007_cycles.csv",
        "B0007_tail15_PRIMARY": "B0007_tail15_PRIMARY_cycles.csv",
    }
    report = {}
    for key, filename in files.items():
        recalculated = metrics_from_csv(CONFIG.reference_dir / filename)
        expected = stored[key]
        deltas = {
            name: abs(float(recalculated[name]) - float(expected[name]))
            for name in ("r2", "rmse_percent_points", "mae_percent_points")
        }
        passed = recalculated["n_cycles"] == expected["n_cycles"] and max(
            deltas.values()
        ) < 1e-12
        report[key] = {"passed": passed, "deltas": deltas}
        print(
            f"{key:30s} R2={recalculated['r2_percent']:10.6f}% "
            f"RMSE={recalculated['rmse_percent_points']:.6f} pp "
            f"{'OK' if passed else 'FAIL'}"
        )
    if not all(item["passed"] for item in report.values()):
        raise AssertionError("Los CSV historicos no coinciden con metrics.json")
    return report


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_run_metadata(history: list[dict]) -> None:
    """Deja configuracion, entorno, hashes e historial junto a la corrida."""

    CONFIG.output_dir.mkdir(parents=True, exist_ok=True)
    (CONFIG.output_dir / "resolved_config.json").write_text(
        json.dumps(asdict(CONFIG), indent=2, sort_keys=True), encoding="utf-8"
    )
    provenance = {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(DEVICE),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "raw_sha256": {
            battery_id: sha256(CONFIG.raw_dir / f"{battery_id}.mat")
            for battery_id in (
                CONFIG.data.train_battery,
                *CONFIG.data.test_batteries,
            )
        },
    }
    (CONFIG.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(CONFIG.output_dir / "history.csv", history)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-reference",
        action="store_true",
        help="recalcula las metricas historicas incluidas y termina",
    )
    parser.add_argument("--resume", action="store_true", help="reanuda checkpoint_last.pt")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permite una prueba funcional en CPU; no replica el entorno L40S",
    )
    args = parser.parse_args()

    if args.verify_reference:
        verify_reference()
        return
    if DEVICE.type != "cuda" and not args.allow_cpu:
        raise RuntimeError(
            "La corrida de referencia uso CUDA/L40S. Use --allow-cpu solo para "
            "una prueba funcional, no para comparar igualdad numerica."
        )

    model, history = train_model(resume=args.resume)
    write_run_metadata(history)
    result = evaluate_model(model)
    print("\nResumen de curva completa (diagnostico):")
    for key in ("B0005_full_DIAGNOSTIC_ONLY", "B0006", "B0007"):
        metric = result["metrics"][key]
        print(
            f"{key:30s} R2={metric['r2_percent']:9.4f}% "
            f"RMSE={metric['rmse_percent_points']:.4f} pp"
        )


if __name__ == "__main__":
    main()

