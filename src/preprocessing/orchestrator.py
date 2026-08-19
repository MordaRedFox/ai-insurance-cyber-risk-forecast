"""
Оркестратор препроцессинга данных
"""

import pandas as pd
from pathlib import Path
from typing import Union, Optional
from src.utils import save_dataframe, save_pickle, load_pickle
from src.preprocessing.loader import DataLoader
from src.preprocessing.feature_engineering import FeatureEngineer
from src.preprocessing.encoding import CategoryEncoder
from src.preprocessing.scaling import FeatureScaler
from src.preprocessing.target_builder import TargetBuilder


class DataPreprocessor:
    """
    Главный класс, управляющий всеми этапами препроцессинга
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Инициализирует препроцессор с заданной конфигурацией
        Args:
            config: словарь конфигурации (опционально)
        """
        self.config = config or {}
        self.loader = DataLoader('')
        self.feature_engineer = FeatureEngineer(self.config)
        self.encoder = CategoryEncoder(self.config)
        self.scaler = FeatureScaler(self.config, exclude_cols=[
            'it_density', 'sec_effectiveness', 'downtime_cap'])
        self.target_builder = TargetBuilder(quantile=0.90)

    def process(self, input_path: Union[str, Path],
                output_path: Union[str, Path], save_scaler: bool = True,
                scaler_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Выполняет полный цикл препроцессинга
        Args:
            input_path: путь к сырым данным
            output_path: путь для сохранения обработанных данных
            save_scaler: сохранять ли обученный scaler
            scaler_path: путь для сохранения scaler (если не указан,
                используется output_path с суффиксом _scaler.pkl)
        Returns:
            Обработанный DataFrame
        """
        self.loader.input_path = Path(input_path)
        df = self.loader.load()
        print(f"Загружено {len(df)} записей")
        df = self.feature_engineer.fit_transform(df)
        print("Инженерные признаки добавлены")
        df = self.encoder.fit_transform(df)
        print("Категориальные признаки закодированы")
        df = self.scaler.fit_transform(df)
        print("Признаки масштабированы")
        df = self.target_builder.build(df)
        print("Целевые переменные созданы")
        save_dataframe(df, output_path)
        print(f"Обработанные данные сохранены в {output_path}")
        if save_scaler:
            scaler_path = scaler_path or Path(
                output_path).with_suffix('_scaler.pkl')
            preprocessor_params = {
                'scaler': self.scaler.get_scaler_params(),
                'encoder': self.encoder.encoder,
                'categories': self.encoder.categories
            }
            save_pickle(preprocessor_params, scaler_path)
            print(f"Параметры препроцессора сохранены в {scaler_path}")
        return df

    def load_scaler(self, scaler_path: Union[str, Path]) -> None:
        """
        Загружает сохраненный scaler вместе со списком колонок
        Args:
            scaler_path: путь к файлу с параметрами scaler
        """
        params = load_pickle(scaler_path)
        self.scaler.set_scaler_params(params)
        print(f"Скалер загружен из {scaler_path}")
