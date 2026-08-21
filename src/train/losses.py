"""
Функции потерь, используемые при обучении
"""

import torch


def pinball_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                 tau: float = 0.9) -> torch.Tensor:
    """
    Вычисляет квантильную потерю (Pinball Loss) для заданного уровня tau
    Args:
        y_pred: предсказанные значения
        y_true: истинные значения
        tau: уровень квантиля (0..1)
    Returns:
        Скалярное значение потери
    """
    errors = y_true - y_pred
    loss = torch.where(errors >= 0, tau * errors, (tau - 1) * errors)
    return loss.mean()
