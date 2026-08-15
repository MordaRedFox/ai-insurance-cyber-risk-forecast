"""
Генератор долевых признаков (бета-распределение, зависящее от отрасли)
"""

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from src.data_generation.base import BaseGenerator


class ShareFeatureGenerator(BaseGenerator):
    """
    Генерирует доли online_share, cloud_share, training_percent
    """
    SHARE_NAMES = ['online_share', 'cloud_share', 'training_percent']

    def generate(self, industries: np.ndarray) -> pd.DataFrame:
        """
        Генерирует доли online_share, cloud_share, training_percent
        Args:
            industries: массив отраслей размером (num_samples,)
        Returns:
            DataFrame с колонками из SHARE_NAMES
        """
        beta_params = self.config['beta_params']
        df_shares = pd.DataFrame()
        for share_name in self.SHARE_NAMES:
            shares = np.zeros(self.num_samples)
            for ind in self.industry_names:
                mask = (industries == ind)
                alpha = beta_params[share_name][ind]['alpha']
                beta = beta_params[share_name][ind]['beta']
                shares[mask] = beta_dist.rvs(
                    alpha, beta, size=np.sum(mask), random_state=self.rng)
            df_shares[share_name] = shares
        return df_shares
