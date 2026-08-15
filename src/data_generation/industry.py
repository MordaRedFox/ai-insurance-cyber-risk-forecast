"""
Генератор отраслей
"""

import numpy as np
from src.data_generation.base import BaseGenerator


class IndustryGenerator(BaseGenerator):
    """
    Генерирует массив отраслей согласно заданным вероятностям
    """

    def generate(self) -> np.ndarray:
        """
        Генерирует массив отраслей согласно заданным вероятностям
        Returns:
            Массив строк с названиями отраслей размером (num_samples,)
        """
        return self.rng.choice(
            self.industry_names,
            size=self.num_samples,
            p=list(self.config['industry_probs'].values())
        )
