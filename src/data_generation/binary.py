"""
Генератор бинарных признаков с вероятностью, зависящей от отрасли
"""

import numpy as np
import pandas as pd
from src.data_generation.base import BaseGenerator


class BinaryFeatureGenerator(BaseGenerator):
    """
    Генерирует бинарные признаки (0/1) по отраслевым вероятностям
    """
    BINARY_NAMES = [
        'has_policy', 'has_soc_siem', 'has_irp', 'has_insurance',
        'has_network_segmentation', 'has_eol_software'
    ]

    def generate(self, industries: np.ndarray) -> pd.DataFrame:
        """
        Генерирует бинарные признаки (0/1) по отраслевым вероятностям
        Args:
            industries: массив отраслей размером (num_samples,)
        Returns:
            DataFrame с колонками из BINARY_NAMES
        """
        binary_probs = self.config['binary_probs']
        df_binary = pd.DataFrame()
        for bname in self.BINARY_NAMES:
            probs_by_industry = binary_probs[bname]
            vals = np.zeros(self.num_samples)
            for ind in self.industry_names:
                mask = (industries == ind)
                p = probs_by_industry[ind]
                vals[mask] = self.rng.binomial(1, p, size=np.sum(mask))
            df_binary[bname] = vals
        return df_binary
