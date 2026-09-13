# -*- coding: utf-8 -*-
"""Entrena el modelo base y guarda las curvas de salud de las baterías.

Uso normal::

    python preprocessing.py
    python train.py

Al terminar, guarda una grafica por bateria y el CSV minimo usado para
construirla: ciclo, SoH real y SoH predicho.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

# Configuración de CUDA/cuBLAS para repetir las operaciones numéricas.
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
    """Combina ventanas y tiempos independientes para evaluar la ecuación física."""

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
    """Combina el residuo de Verhulst y el error de la condición inicial."""

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


def prediction_rows(
    model: SA_PI_AGRU, arrays: dict[str, torch.Tensor]
) -> list[dict]:
    """Devuelve solamente los tres valores necesarios para la grafica."""

    predictions = predict_windows(model, arrays)
    cycles, actual, predicted = aggregate_by_cycle(arrays, predictions)
    return [
        {
            "cycle": int(cycle),
            "soh_true": float(truth),
            "soh_pred": float(estimate),
        }
        for cycle, truth, estimate in zip(cycles, actual, predicted)
    ]


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
    # Los estados aleatorios deben cargarse en CPU; modelo y optimizador
    # trasladan sus parámetros al dispositivo al restaurarlos.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch 1.13.1 no conoce weights_only.
        return torch.load(path, map_location="cpu")


def train_model(resume: bool = False) -> tuple[SA_PI_AGRU, list[dict]]:
    """Entrena durante las épocas indicadas, con una ventana por ciclo."""

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
        if state.get("run_name") != CONFIG.run_name or state.get("seed") != CONFIG.train.seed:
            raise ValueError("El checkpoint no corresponde al nombre y semilla configurados")
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
        f"ejecucion={CONFIG.run_name} semilla={CONFIG.train.seed} dispositivo={DEVICE} "
        f"ciclos_entrenamiento={len(torch.unique(train_data['cycle']))}"
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


def evaluate_model(model: SA_PI_AGRU) -> None:
    """Guarda una curva completa y su CSV para cada batería configurada."""

    destination = CONFIG.output_dir / "resultados"
    destination.mkdir(parents=True, exist_ok=True)
    train_id = CONFIG.data.train_battery
    train_battery = load_battery(train_id)
    full_rows: list[dict] = []
    for split in ("train", "val", "test"):
        full_rows.extend(prediction_rows(model, tensor_split(train_battery, split)))
    full_rows.sort(key=lambda row: row["cycle"])
    write_csv(destination / f"{train_id}_full_cycles.csv", full_rows)
    plot_cycles(destination / f"{train_id}.png", train_id, full_rows)

    for battery_id in CONFIG.data.test_batteries:
        arrays = tensor_split(load_battery(battery_id), "test")
        rows = prediction_rows(model, arrays)
        write_csv(destination / f"{battery_id}_cycles.csv", rows)
        plot_cycles(destination / f"{battery_id}.png", battery_id, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="reanuda checkpoint_last.pt")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permite entrenar en CPU si no hay GPU disponible",
    )
    args = parser.parse_args()

    if DEVICE.type != "cuda" and not args.allow_cpu:
        raise RuntimeError(
            "No se detectó una GPU disponible para PyTorch. "
            "Use --allow-cpu para entrenar en CPU."
        )

    model, _ = train_model(resume=args.resume)
    evaluate_model(model)
    print(f"\nGraficas y CSV guardados en: {CONFIG.output_dir / 'resultados'}")


if __name__ == "__main__":
    main()
