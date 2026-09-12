"""
Провайдер предсказаний для визуализации
Реализован как синглтон: инференс запускается один раз,
дальше данные берутся из кэша
"""

from typing import Optional
import pandas as pd
from src.infer import Inference
from src.utils import load_dataframe


class PredictionProvider:
    """
    Загружает и кэширует предсказания модели
    """
    _instance: Optional['PredictionProvider'] = None
    _data: Optional[pd.DataFrame] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self,
                 raw_data_path: str = 'data/presentation/presentation_companies.csv',
                 model_path: str = 'models/checkpoints/best_model.pth',
                 preprocessor_path: str = 'models/scaler.pkl',
                 model_config: str = 'configs/model_config.yaml'):
        """
        Args:
            raw_data_path: путь к презентационному CSV
            model_path: путь к весам модели
            preprocessor_path: путь к scaler.pkl
            model_config: путь к YAML-конфигу модели
        """
        if self._data is not None:
            return
        self._load(
            raw_data_path=raw_data_path,
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            model_config=model_config,
        )

    def _load(self, **kwargs) -> None:
        """
        Загружает данные и выполняет инференс
        """
        df_raw = load_dataframe(kwargs['raw_data_path'])
        inference = Inference(
            model_path=kwargs['model_path'],
            preprocessor_path=kwargs['preprocessor_path'],
            model_config=kwargs['model_config'],
            device='cpu',
        )
        predictions = inference.predict(df_raw)
        self.__class__._data = pd.concat([df_raw, predictions], axis=1)

    @property
    def data(self) -> pd.DataFrame:
        """
        Возвращает копию DataFrame с предсказаниями
        Returns:
            DataFrame с предсказаниями
        """
        return self._data.copy()
