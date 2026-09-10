# -*- coding: utf-8 -*-
"""Configuracion unica de la mejor corrida SA-PI-AGRU.

Este repositorio conserva la division original en cuatro archivos de JP.  Los
valores de abajo no son una nueva busqueda: corresponden a la corrida
C1/B0005/seed 58 terminada en 10 000 actualizaciones.
"""

from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DataConfig:
    """Como se leen y separan las baterias NASA."""

    train_battery: str = "B0005"
    test_batteries: tuple[str, ...] = ("B0006", "B0007")
    nominal_capacity_ah: float = 2.0
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    window_size: int = 50
    stride: int = 1
    drop_zero_rows: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """Arquitectura documentada por JP y concretada en la corrida C1."""

    sensor_count: int = 6
    hidden_size: int = 32
    num_layers: int = 3
    dropout: float = 0.20
    initial_r_normalized: float = 2.0


@dataclass(frozen=True)
class PhysicsConfig:
    """Ecuacion de Verhulst aplicada a u = 1 - SoH."""

    carrying_capacity: float = 0.50  # K
    offset: float = 0.05             # C
    initial_loss: float = 0.10       # u(0)
    initial_condition_weight: float = 1.0
    residual_weight: float = 1.0
    collocation_points: int = 5_000
    collocation_time_max: float = 1.0


@dataclass(frozen=True)
class TrainConfig:
    """Parametros que produjeron el endpoint de referencia."""

    epochs: int = 10_000
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    seed: int = 58
    validation_every: int = 50
    checkpoint_every: int = 500
    log_every: int = 100


@dataclass(frozen=True)
class Config:
    """Agrupa parametros y rutas sin numeros magicos en otros modulos."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    run_name: str = "C1_B0005_seed58"

    @property
    def raw_dir(self) -> Path:
        return ROOT / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return ROOT / "data" / "processed"

    @property
    def output_dir(self) -> Path:
        return ROOT / "outputs" / self.run_name

    @property
    def reference_dir(self) -> Path:
        return ROOT / "reference" / "best_run"

    def validate(self) -> None:
        if self.data.train_battery in self.data.test_batteries:
            raise ValueError("La bateria de entrenamiento no puede ser de prueba")
        if self.data.train_fraction + self.data.validation_fraction >= 1:
            raise ValueError("Los splits train/validation deben dejar un test")
        if self.data.window_size < 2 or self.data.stride < 1:
            raise ValueError("window_size y stride deben ser positivos")
        if self.model.sensor_count != 6:
            raise ValueError("La corrida de referencia usa exactamente 6 sensores")
        if not (
            self.physics.carrying_capacity
            > self.physics.initial_loss
            >= self.physics.offset
            >= 0
        ):
            raise ValueError("Se requiere K > u(0) >= C >= 0")
        if self.train.epochs < 1 or self.train.batch_size < 1:
            raise ValueError("epochs y batch_size deben ser positivos")


CONFIG = Config()
CONFIG.validate()

