"""
Многоголовая нейросеть для прогнозирования киберрисков

Модель имеет общий бэкбон (Shared Backbone) и 3 головы:
- T1: вероятность инцидента (бинарная классификация)
- T2: условный ущерб при инциденте (регрессия, только для y > 0)
- T3: безусловный ущерб (регрессия, для всех)
"""

import torch
import torch.nn as nn
from typing import Dict


class CyberRiskModel(nn.Module):
    """
    Модель для прогнозирования финансовых потерь от киберинцидентов
    """

    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64],
                 dropout: float = 0.3, head_dropout: float = 0.2,
                 activation: str = 'relu', use_batch_norm: bool = True):
        """
        Инициализирует модель
        Args:
            input_dim: размерность входных признаков
            hidden_dims: список размеров скрытых слоев бэкбона
            dropout: вероятность дропаута в бэкбоне
            head_dropout: вероятность дропаута перед головами
            activation: тип активации ('relu' или 'leaky_relu')
            use_batch_norm: использовать ли BatchNorm в бэкбоне
        """
        super().__init__()
        # Выбор функции активации
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        else:
            self.activation = nn.ReLU()
        # Построение бэкбона
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        # Общий дропаут перед головами
        self.head_dropout = nn.Dropout(head_dropout)
        # Головы
        self.head_t1 = nn.Linear(prev_dim, 1)
        self.head_t2 = nn.Linear(prev_dim, 1)
        self.head_t3 = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Прямой проход модели
        Args:
            x: входной тензор размерности (batch_size, input_dim)
        Returns:
            Словарь с выходами голов (везде shape: [batch, 1]):
                - 't1': логиты вероятности инцидента
                - 't2': предсказанный логарифм условного ущерба
                - 't3': предсказанный логарифм безусловного ущерба
        """
        features = self.backbone(x)
        features = self.head_dropout(features)
        return {
            't1': self.head_t1(features),
            't2': self.head_t2(features),
            't3': self.head_t3(features)
        }

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Возвращает вероятность инцидента (сигмоид от t1)
        Args:
            x: входной тензор
        Returns:
            Тензор вероятностей (shape: [batch, 1])
        """
        return torch.sigmoid(self.forward(x)['t1'])
