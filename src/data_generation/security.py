"""
Генератор расходов на кибербезопасность как доля от выручки
"""

import numpy as np
from src.data_generation.base import BaseGenerator


class SecuritySpendGenerator(BaseGenerator):
    """
    Генерирует расходы на безопасность на основе выручки и
    отраслевого коэффициента
    """

    def generate(self, revenue: np.ndarray,
                 industries: np.ndarray) -> np.ndarray:
        """
        Генерирует расходы на безопасность на основе выручки и
        отраслевого коэффициента
        Args:
            revenue: массив выручки (num_samples,)
            industries: массив отраслей (num_samples,)
        Returns:
            Массив расходов на безопасность
        """
        security_cfg = self.config['security_spend']
        base_share_dict = security_cfg['base_share']
        noise_sigma = security_cfg['noise_sigma']
        base_share = np.array([base_share_dict[ind] for ind in industries])
        noise = self.rng.lognormal(
            mean=0, sigma=noise_sigma, size=self.num_samples)
        security_spend = revenue * base_share * noise
        security_spend = np.maximum(security_spend, 0)
        return security_spend
