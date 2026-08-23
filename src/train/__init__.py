"""
Пакет для обучения модели
"""

from src.train.trainer import Trainer
from src.train.dataloader import prepare_dataloaders

__all__ = [
    'Trainer',
    'prepare_dataloaders',
]
