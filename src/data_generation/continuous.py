"""
Генератор непрерывных признаков с лог-нормальным распределением и корреляцией
"""

import numpy as np
import pandas as pd
import logging
from scipy.stats import multivariate_normal
from src.data_generation.base import BaseGenerator

logger = logging.getLogger(__name__)


class ContinuousFeatureGenerator(BaseGenerator):
    """
    Генерирует непрерывные признаки, учитывая корреляционную матрицу и
    маскирование нулей
    """
    FEATURE_NAMES = [
        'revenue', 'intangible_assets', 'employees', 'servers',
        'critical_apps', 'iot_ot', 'personal_data',
        'incidents_3y', 'past_loss', 'past_payout'
    ]

    def generate(self) -> pd.DataFrame:
        """
        Генерирует непрерывные признаки, учитывая корреляционную матрицу и
        маскирование нулей
        Returns:
            DataFrame с колонками из FEATURE_NAMES
        """
        log_params = self.config['log_normal_params']
        means = np.array(
            [log_params[name]['mean'] for name in self.FEATURE_NAMES])
        stds = np.array(
            [log_params[name]['std'] for name in self.FEATURE_NAMES])
        # Извлечение корреляционной матрицы из конфига
        corr_values = list(self.config['correlation_matrix'].values())
        rows_have_same_length = all(
            len(row) == len(corr_values) for row in corr_values)
        assert rows_have_same_length, "Строки имеют разную длину"
        corr_matrix = np.array(corr_values)
        n_features = len(self.FEATURE_NAMES)
        assert corr_matrix.shape == (n_features, n_features), \
            "Несоответствие формы корреляционной матрицы"
        # Ковариационная матрица с регуляризацией
        cov_matrix = np.diag(stds) @ corr_matrix @ np.diag(stds)
        eps = 1e-6
        while True:
            try:
                np.linalg.cholesky(
                    cov_matrix + eps * np.eye(cov_matrix.shape[0]))
                break
            except np.linalg.LinAlgError:
                eps *= 10
                logger.warning(f"Увеличение регуляризации эпсилон до {eps}")
        cov_matrix += eps * np.eye(cov_matrix.shape[0])
        mvn = multivariate_normal(
            mean=means, cov=cov_matrix, seed=self.config['seed'])
        log_values = mvn.rvs(size=self.num_samples)
        values = np.exp(log_values)
        values = np.clip(values, 0, 1e9)
        values = np.nan_to_num(values, nan=0.0)
        df_cont = pd.DataFrame(values, columns=self.FEATURE_NAMES)
        # Маскирование нулями для указанных признаков
        zero_probs = self.config.get('zero_probabilities', {})
        for col_name, prob in zero_probs.items():
            if col_name in df_cont.columns:
                mask = self.rng.binomial(
                    1, prob, size=self.num_samples).astype(bool)
                df_cont.loc[mask, col_name] = 0.0
        return df_cont
