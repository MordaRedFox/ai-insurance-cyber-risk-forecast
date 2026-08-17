"""
Создание инженерных признаков
"""

import pandas as pd
import numpy as np
from src.preprocessing.base import BasePreprocessor


class FeatureEngineer(BasePreprocessor):
    """
    Создает инженерные признаки: it_density, sec_effectiveness, downtime_cap
    Также выполняет очистку данных: заменяет inf и NaN на 0 во всех
    числовых колонках
    """

    def __init__(self, config: dict = None):
        """
        Инициализирует инженера признаков
        Args:
            config: словарь конфигурации (опционально)
        """
        super().__init__(config)
        self.feature_names = [
            'it_density', 'sec_effectiveness', 'downtime_cap']

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет инженерные признаки к DataFrame и очищает данные
        Args:
            X: исходный DataFrame с колонками на русском языке
        Returns:
            DataFrame с добавленными колонками:
                - it_density = серверы / сотрудники
                - sec_effectiveness = расходы на безопасность / выручка
                - downtime_cap = потери от часа простоя * MTD
            Все NaN и inf в новых колонках заменяются на 0.
        """
        df = X.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].fillna(0)
        # it_density = серверы / сотрудники
        servers_col = 'Количество серверов'
        employees_col = 'Количество сотрудников'
        if servers_col in df.columns and employees_col in df.columns:
            df['it_density'] = df[servers_col] / (df[employees_col] + 1e-6)
        # sec_effectiveness = расходы на безопасность / выручка
        spend_col = 'Расходы на кибербезопасность за последний год (млн руб.)'
        revenue_col = 'Годовая выручка (млн руб.)'
        if spend_col in df.columns and revenue_col in df.columns:
            df['sec_effectiveness'] = df[spend_col] / (df[revenue_col] + 1e-6)
        # downtime_cap = потери от часа простоя * MTD
        downtime_loss_col = 'Потери от часа простоя (млн руб.)'
        mtd_col = 'Максимально допустимое время простоя критических сервисов (в часах)'
        if downtime_loss_col in df.columns and mtd_col in df.columns:
            df['downtime_cap'] = df[downtime_loss_col] * df[mtd_col]
        for col in self.feature_names:
            if col in df.columns:
                df[col] = df[col].fillna(0)
                df[col] = df[col].replace([np.inf, -np.inf], 0)
        return df
