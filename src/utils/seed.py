"""
Утилиты для установки и настройки seed
"""

import random
import torch
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Фиксирует случайный seed для воспроизводимости
    Args:
        seed: целое число для инициализации всех генераторов случайных чисел
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
