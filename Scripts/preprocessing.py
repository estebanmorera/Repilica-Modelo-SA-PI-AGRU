# -*- coding: utf-8 -*-
"""Convierte los .mat de NASA en ventanas limpias para SA-PI-AGRU.

La regla mas importante es separar ciclos completos antes de crear ventanas.
Asi ninguna secuencia cruza una frontera fisica ni aparece en dos splits.  El
escalador se ajusta solo con B0005/train y la capacidad se usa exclusivamente
para construir el objetivo u = 1 - SoH; nunca entra como sensor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio

from config import CONFIG


SENSOR_NAMES = (
    "voltage_measured",
    "current_measured",
    "temperature_measured",
    "ambient_temperature",
    "current_load",
    "voltage_load",
)


def _vector(value) -> np.ndarray:
    """Devuelve cualquier campo MATLAB como vector float64 de una dimension."""

    return np.atleast_1d(value).astype(np.float64).reshape(-1)


def extract_discharge_cycles(mat_path: Path, battery_id: str) -> list[dict]:
    """Extrae sensores y capacidad de cada ciclo de descarga valido."""

    if not mat_path.exists():
        raise FileNotFoundError(f"Falta el dataset: {mat_path}")

    matlab = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if battery_id not in matlab:
        raise KeyError(f"{mat_path.name} no contiene la estructura {battery_id}")

    cycles: list[dict] = []
    discharge_id = 0
    for raw_cycle in np.atleast_1d(matlab[battery_id].cycle):
        if str(raw_cycle.type).strip().lower() != "discharge":
            continue
        discharge_id += 1
        data = raw_cycle.data
        series = (
            _vector(data.Voltage_measured),
            _vector(data.Current_measured),
            _vector(data.Temperature_measured),
            _vector(data.Current_load),
            _vector(data.Voltage_load),
        )
        sample_count = min(map(len, series))
        ambient = np.full(sample_count, float(raw_cycle.ambient_temperature))
        sensors = np.column_stack(
            (
                series[0][:sample_count],
                series[1][:sample_count],
                series[2][:sample_count],
                ambient,
                series[3][:sample_count],
                series[4][:sample_count],
            )
        )
        capacity = float(_vector(data.Capacity)[0])

        valid = np.isfinite(sensors).all(axis=1)
        if CONFIG.data.drop_zero_rows:
            valid &= np.all(sensors != 0.0, axis=1)
        sensors = sensors[valid]
        if np.isfinite(capacity) and capacity > 0 and len(sensors) >= 2:
            cycles.append(
                {"cycle": discharge_id, "sensors": sensors, "capacity": capacity}
            )

    if not cycles:
        raise ValueError(f"No se extrajeron ciclos validos de {mat_path}")
    return cycles


def temporal_split(cycles: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Divide 70/15/15 en orden temporal usando ciclos completos."""

    count = len(cycles)
    if count < 3:
        raise ValueError("Se requieren al menos tres ciclos")
    train_end = min(max(1, int(count * CONFIG.data.train_fraction)), count - 2)
    validation_end = min(
        max(
            train_end + 1,
            int(
                count
                * (CONFIG.data.train_fraction + CONFIG.data.validation_fraction)
            ),
        ),
        count - 1,
    )
    return (
        cycles[:train_end],
        cycles[train_end:validation_end],
        cycles[validation_end:],
    )


def fit_scaler(cycles: Iterable[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Ajusta minimos y maximos solo con los sensores de train."""

    values = np.concatenate([cycle["sensors"] for cycle in cycles], axis=0)
    return values.min(axis=0), values.max(axis=0)


def apply_scaler(
    values: np.ndarray, minimum: np.ndarray, maximum: np.ndarray
) -> np.ndarray:
    scale = maximum - minimum
    return (values - minimum) / np.where(scale == 0, 1.0, scale)


def _resample(values: np.ndarray, length: int) -> np.ndarray:
    """Solo se usa si un ciclo es mas corto que la ventana solicitada."""

    old_positions = np.linspace(0.0, 1.0, len(values))
    new_positions = np.linspace(0.0, 1.0, length)
    return np.column_stack(
        [
            np.interp(new_positions, old_positions, values[:, column])
            for column in range(values.shape[1])
        ]
    )


def cycle_windows(
    cycle: dict,
    minimum: np.ndarray,
    maximum: np.ndarray,
    time_bounds: tuple[float, float],
) -> dict[str, np.ndarray]:
    """Crea todas las ventanas dentro de un solo ciclo."""

    features = apply_scaler(cycle["sensors"], minimum, maximum)
    window = CONFIG.data.window_size
    if len(features) < window:
        sequences = _resample(features, window)[None, ...]
    else:
        starts = np.arange(0, len(features) - window + 1, CONFIG.data.stride)
        sequences = np.stack([features[start : start + window] for start in starts])

    count = len(sequences)
    cycle_id = float(cycle["cycle"])
    time_min, time_max = time_bounds
    time_value = (cycle_id - time_min) / max(time_max - time_min, 1.0)
    soh = float(cycle["capacity"]) / CONFIG.data.nominal_capacity_ah
    return {
        "X": sequences.astype(np.float32),
        "t": np.full((count, 1), time_value, dtype=np.float32),
        "u": np.full((count, 1), 1.0 - soh, dtype=np.float32),
        "cycle": np.full((count, 1), cycle_id, dtype=np.float32),
        "weight": np.full((count, 1), 1.0 / count, dtype=np.float32),
    }


def build_arrays(
    cycles: list[dict],
    minimum: np.ndarray,
    maximum: np.ndarray,
    time_bounds: tuple[float, float],
) -> dict[str, np.ndarray]:
    blocks = [
        cycle_windows(cycle, minimum, maximum, time_bounds) for cycle in cycles
    ]
    return {
        key: np.concatenate([block[key] for block in blocks], axis=0)
        for key in blocks[0]
    }


def _metadata(
    battery_id: str,
    minimum: np.ndarray,
    maximum: np.ndarray,
    time_bounds: tuple[float, float],
    splits: dict[str, list[dict]],
) -> dict:
    return {
        "schema": "cycle_disjoint_v2",
        "battery": battery_id,
        "feature_names": list(SENSOR_NAMES),
        "feature_mode": "paper_core_no_capacity",
        "sequence_mode": "within_cycle_windows",
        "drop_zero_rows": CONFIG.data.drop_zero_rows,
        "external_scaler": "train_only",
        "nominal_capacity_ah": CONFIG.data.nominal_capacity_ah,
        "scaler_min": minimum.tolist(),
        "scaler_max": maximum.tolist(),
        "time_cycle_min": float(time_bounds[0]),
        "time_cycle_max": float(time_bounds[1]),
        "cycle_ids": {
            name: [int(cycle["cycle"]) for cycle in values]
            for name, values in splits.items()
        },
    }


def _save_npz(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def process_all() -> dict:
    """Procesa B0005 primero y reutiliza su scaler en B0006/B0007."""

    CONFIG.validate()
    train_id = CONFIG.data.train_battery
    train_cycles = extract_discharge_cycles(
        CONFIG.raw_dir / f"{train_id}.mat", train_id
    )
    train, validation, test = temporal_split(train_cycles)
    minimum, maximum = fit_scaler(train)
    time_bounds = (float(train[0]["cycle"]), float(train[-1]["cycle"]))
    split_cycles = {"train": train, "val": validation, "test": test}

    train_arrays: dict[str, np.ndarray] = {}
    for split_name, cycles in split_cycles.items():
        block = build_arrays(cycles, minimum, maximum, time_bounds)
        train_arrays.update(
            {f"{key}_{split_name}": value for key, value in block.items()}
        )
    _save_npz(
        CONFIG.processed_dir / f"{train_id}.npz",
        train_arrays,
        _metadata(train_id, minimum, maximum, time_bounds, split_cycles),
    )

    report = {
        train_id: {
            "cycles": {name: len(values) for name, values in split_cycles.items()},
            "windows": {
                name: int(len(train_arrays[f"X_{name}"])) for name in split_cycles
            },
        }
    }

    for battery_id in CONFIG.data.test_batteries:
        cycles = extract_discharge_cycles(
            CONFIG.raw_dir / f"{battery_id}.mat", battery_id
        )
        block = build_arrays(cycles, minimum, maximum, time_bounds)
        arrays = {f"{key}_test": value for key, value in block.items()}
        _save_npz(
            CONFIG.processed_dir / f"{battery_id}.npz",
            arrays,
            _metadata(
                battery_id,
                minimum,
                maximum,
                time_bounds,
                {"test": cycles},
            ),
        )
        report[battery_id] = {
            "cycles": {"test": len(cycles)},
            "windows": {"test": int(len(block["X"]))},
        }

    manifest = {
        "description": "Datos exactos para C1/B0005/seed58",
        "report": report,
    }
    CONFIG.processed_dir.mkdir(parents=True, exist_ok=True)
    (CONFIG.processed_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    manifest = process_all()
    for battery_id, details in manifest["report"].items():
        print(battery_id, details["cycles"], details["windows"])


if __name__ == "__main__":
    main()

