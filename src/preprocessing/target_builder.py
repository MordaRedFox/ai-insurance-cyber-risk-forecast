"""
Формирование целевых переменных для многоголовой модели
"""

import pandas as pd
import numpy as np


class TargetBuilder:
    """
    Создает целевые переменные T1-T3 на основе Y0:
        - T1: бинарная метка (1 если Y0 > 0)
        - T2: условный ущерб (логарифм Y0 для Y0 > 0, NaN для остальных)
        - T3: безусловный ущерб (логарифм Y0)
    """

    def __init__(self, quantile: float = 0.90):
        """
        Инициализирует построитель целевых переменных
        Args:
            quantile: уровень квантиля для VaR (по умолчанию 0.90)
        """
        self.quantile = quantile

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет колонки T1-T3 в DataFrame
        Args:
            df: DataFrame с колонкой 'Фактический совокупный ущерб за
                последние 12 месяцев (млн руб.)'
        Returns:
            DataFrame с добавленными колонками T1, T2, T3
        """
        y_col = 'Фактический совокупный ущерб за последние 12 месяцев (млн руб.)'
        df = df.copy()
        df['T1'] = (df[y_col] > 0).astype(float)
        df['T2'] = np.log1p(df[y_col])
        df.loc[df[y_col] == 0, 'T2'] = np.nan
        df['T3'] = np.log1p(df[y_col])
        return df

    def get_target_columns(self):
        """
        Возвращает имена целевых колонок
        Returns:
            Список строк ['T1', 'T2', 'T3']
        """
        return ['T1', 'T2', 'T3']
