# -*- coding: utf-8 -*-
"""GRU con atención y ecuación física para estimar la salud de baterías."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CONFIG


def inverse_softplus(value: float) -> float:
    """Valor sin transformar que produce ``value`` despues de softplus."""

    if value <= 0:
        raise ValueError("La tasa r inicial debe ser positiva")
    return math.log(math.expm1(value))


class AGRU(nn.Module):
    """GRU apilada seguida de atencion scaled dot-product."""

    def __init__(self) -> None:
        super().__init__()
        model = CONFIG.model
        self.gru = nn.GRU(
            input_size=model.sensor_count,
            hidden_size=model.hidden_size,
            num_layers=model.num_layers,
            batch_first=True,
            dropout=model.dropout,
        )

    def forward(self, sensors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, _ = self.gru(sensors)
        query = sequence[:, -1:, :]
        scores = torch.bmm(sequence, query.transpose(1, 2))
        scores = scores / math.sqrt(CONFIG.model.hidden_size)
        attention = F.softmax(scores, dim=1)
        context = torch.sum(sequence * attention, dim=1)
        return context, attention.squeeze(-1)


class SA_PI_AGRU(nn.Module):
    """Predice la perdida de capacidad ``u = 1 - SoH`` y aprende ``r > 0``."""

    def __init__(self) -> None:
        super().__init__()
        hidden = CONFIG.model.hidden_size
        self.agru = AGRU()
        self.head = nn.Sequential(
            nn.Linear(hidden + 1, hidden),  # contexto + numero de ciclo normalizado
            nn.GELU(),
            nn.Dropout(CONFIG.model.dropout),
            nn.Linear(hidden, 1),           # salida lineal; Softmax-1 seria siempre 1
        )
        self.r_raw = nn.Parameter(
            torch.tensor(
                inverse_softplus(CONFIG.model.initial_r_normalized),
                dtype=torch.float32,
            )
        )

    def forward(self, sensors: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        context, _ = self.agru(sensors)
        return self.head(torch.cat((context, time), dim=1))

    @property
    def r(self) -> torch.Tensor:
        """Tasa de Verhulst positiva en tiempo normalizado."""

        return F.softplus(self.r_raw)
