"""
One-Hot кодирование категориальных признаков
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from src.preprocessing.base import BasePreprocessor


class CategoryEncoder(BasePreprocessor):
    """
    One-Hot кодирует колонку 'Отрасль'
    Обучает OneHotEncoder на колонке 'Отрасль' и заменяет ее на набор бинарных
    колонок с префиксом 'industry_'
    """

    def __init__(self, config: dict = None):
        """
        Инициализирует энкодер
        Args:
            config: словарь конфигурации (опционально)
        """
        super().__init__(config)
        self.encoder = OneHotEncoder(
            sparse_output=False, handle_unknown='ignore')
        self.categories = None

    def fit(self, X: pd.DataFrame):
        """
        Обучает OneHotEncoder на колонке 'Отрасль'
        Args:
            X: DataFrame с колонкой 'Отрасль'
        Returns:
            self
        """
        industry_col = 'Отрасль'
        if industry_col in X.columns:
            self.encoder.fit(X[[industry_col]])
            self.categories = self.encoder.categories_[0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет One-Hot кодирование к колонке 'Отрасль'
        Args:
            X: DataFrame с колонкой 'Отрасль'
        Returns:
            DataFrame с удаленной колонкой 'Отрасль' и добавленными бинарными
            колонками industry_*
        """
        df = X.copy()
        industry_col = 'Отрасль'
        if industry_col in df.columns:
            encoded = self.encoder.transform(df[[industry_col]])
            col_names = [f'industry_{cat}' for cat in self.categories]
            encoded_df = pd.DataFrame(
                encoded, columns=col_names, index=df.index)
            df = pd.concat([df, encoded_df], axis=1)
            df.drop(columns=[industry_col], inplace=True)
        return df

    def get_feature_names(self):
        """
        Возвращает имена созданных one-hot колонок
        Returns:
            Список строк с именами колонок или пустой список,
            если энкодер не обучен
        """
        if self.categories is not None:
            return [f'industry_{cat}' for cat in self.categories]
        return []
