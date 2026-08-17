"""
Базовый класс для всех препроцессоров
"""

import pandas as pd
from typing import Dict, Any, Optional


class BasePreprocessor:
    """
    Общий предок для всех препроцессоров, содержащий конфиг и параметры
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализирует препроцессор
        Args:
            config: словарь конфигурации (опционально)
        """
        self.config = config

    def fit(self, X: pd.DataFrame):
        """
        Обучает препроцессор на данных (по умолчанию ничего не делает)
        Args:
            X: входной DataFrame
        """
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет преобразование к данным (по умолчанию возвращает копию)
        Args:
            X: входной DataFrame
        Returns:
            Преобразованный DataFrame
        """
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Обучает препроцессор и применяет преобразование к данным
        Args:
            X: входной DataFrame
        Returns:
            Преобразованный DataFrame
        """
        self.fit(X)
        return self.transform(X)
