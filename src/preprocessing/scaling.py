"""
Масштабирование числовых признаков (StandardScaler)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.preprocessing.base import BasePreprocessor


class FeatureScaler(BasePreprocessor):
    """
    Масштабирует все числовые колонки (кроме бинарных и целевых) с
    помощью StandardScaler
    Исключает из масштабирования:
        - бинарные признаки (0/1)
        - целевую переменную Y0
        - колонки, переданные в exclude_cols
    """

    def __init__(self, config: dict = None, exclude_cols: list = None):
        """
        Инициализирует скейлер
        Args:
            config: словарь конфигурации (опционально)
            exclude_cols: список имен колонок, которые не нужно масштабировать
        """
        super().__init__(config)
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.exclude_cols = exclude_cols or []

    def fit(self, X: pd.DataFrame):
        """
        Обучает StandardScaler на выбранных числовых колонках
        Args:
            X: DataFrame для обучения
        Returns:
            self
        """
        binary_cols = [
            'Наличие формальной политики безопасности',
            'Наличие SOC / SIEM-системы и 24/7 мониторинга',
            'Наличие плана реагирования на инциденты (IRP)',
            'Наличие страхового покрытия киберрисков в прошлом',
            'Наличие сегментации сети',
            'Наличие устаревшего ПО/EOL'
        ]
        target_cols = [
            'Фактический совокупный ущерб за последние 12 месяцев (млн руб.)']
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        exclude_set = set(binary_cols + target_cols + self.exclude_cols)
        self.feature_columns = [
            col for col in numeric_cols if col not in exclude_set]
        if self.feature_columns:
            self.scaler.fit(X[self.feature_columns])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет масштабирование к выбранным колонкам
        Args:
            X: DataFrame для преобразования
        Returns:
            DataFrame с масштабированными колонками
        """
        df = X.copy()
        if self.feature_columns:
            scaled_values = self.scaler.transform(df[self.feature_columns])
            df[self.feature_columns] = scaled_values
        return df

    def get_feature_names(self):
        """
        Возвращает имена масштабируемых колонок
        Returns:
            Список строк с именами колонок
        """
        return self.feature_columns

    def get_scaler(self):
        """
        Возвращает обученный scaler для сохранения
        Returns:
            Объект StandardScaler
        """
        return self.scaler

    def get_scaler_params(self):
        """
        Возвращает параметры scaler и список колонок для сохранения
        Returns:
            Словарь с ключами 'scaler' и 'feature_columns'
        """
        return {
            'scaler': self.scaler, 'feature_columns': self.feature_columns}

    def set_scaler_params(self, params: dict):
        """
        Устанавливает scaler и список колонок из загруженных параметров
        Args:
            params: словарь с ключами 'scaler' и 'feature_columns'
        """
        self.scaler = params['scaler']
        self.feature_columns = params['feature_columns']
