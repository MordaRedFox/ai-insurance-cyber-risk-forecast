"""
Подготовка DataLoader'ов для обучения
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List


def prepare_dataloaders(df: pd.DataFrame, feature_cols: List[str],
                        target_cols: List[str], batch_size: int,
                        val_split: float = 0.2, random_state: int = 42
                        ) -> Tuple[DataLoader, DataLoader]:
    """
    Подготавливает DataLoader'ы для обучения и валидации
    Args:
        df: DataFrame с признаками и целями
        feature_cols: список имен колонок-признаков
        target_cols: список имен целевых колонок (T1, T2, T3, T4)
        batch_size: размер батча
        val_split: доля валидационной выборки
        random_state: seed для разделения
    Returns:
        Кортеж (train_loader, val_loader)
    """
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * val_split)
    train_idx, val_idx = indices[split_idx:], indices[:split_idx]
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_val = torch.tensor(X_val)
    y_val = torch.tensor(y_val)
    train_dataset = TensorDataset(
        X_train, y_train[:, 0], y_train[:, 1], y_train[:, 2], y_train[:, 3])
    val_dataset = TensorDataset(
        X_val, y_val[:, 0], y_val[:, 1], y_val[:, 2], y_val[:, 3])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
