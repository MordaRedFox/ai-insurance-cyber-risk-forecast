"""
Базовый класс для всех генераторов признаков
"""

import numpy as np
from typing import Dict, Any


class BaseGenerator:
    """
    Общий предок для всех генераторов, содержащий конфиг и генератор
    случайных чисел
    """

    def __init__(self, config: Dict[str, Any], rng: np.random.RandomState):
        """
        Args:
            config: словарь конфигурации (загруженный из YAML)
            rng: объект RandomState для воспроизводимости
        """
        self.config = config
        self.rng = rng
        self.num_samples = config['num_samples']
        self.industry_names = list(config['industry_probs'].keys())
