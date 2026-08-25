"""
Инференс модели на новых данных
Выполняет полный пайплайн: загрузка модели, scaler, encoder, препроцессинг
и прогнозирование T1, T2, T3 (с сигмоидом для T1)
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from src.model import CyberRiskModel
from src.preprocessing import DataPreprocessor
from src.utils import load_config, load_pickle


class Inference:
    """
    Класс для инференса обученной модели (3 головы: T1, T2, T3)
    """

    def __init__(self, model_path: Union[str, Path],
                 preprocessor_path: Union[str, Path],
                 model_config: Union[str, Path], device: str = 'cpu'):
        """
        Инициализирует инференс, загружая модель и препроцессор
        Args:
            model_path: путь к файлу с весами модели (.pth)
            preprocessor_path: путь к файлу с параметрами препроцессора (.pkl)
            model_config: путь к YAML с конфигурацией модели
            device: устройство ('cuda' или 'cpu')
        """
        self.device = torch.device(
            device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.model_config = load_config(model_config)
        self.load_preprocessor()
        self.model = None
        self.input_dim = None

    def load_preprocessor(self):
        """
        Загружает scaler, encoder, категории и полный список признаков
        """
        params = load_pickle(self.preprocessor_path)
        self.scaler = params['scaler']['scaler']
        self.feature_columns = params['scaler']['feature_columns']
        self.encoder = params['encoder']
        self.categories = params.get('categories', None)
        self.full_feature_columns = params.get(
            'full_feature_columns', self.feature_columns)

    def load_model(self, input_dim: int):
        """
        Загружает модель с заданной размерностью
        Args:
             input_dim: размерность модели
        """
        hidden_dims = self.model_config['model']['hidden_dims']
        dropout = self.model_config['model']['dropout']
        head_dropout = self.model_config['model'].get('head_dropout', 0.2)
        activation = self.model_config['model'].get('activation', 'relu')
        use_batch_norm = self.model_config['model'].get('use_batch_norm', True)
        self.model = CyberRiskModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            head_dropout=head_dropout,
            activation=activation,
            use_batch_norm=use_batch_norm
        ).to(self.device)
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.input_dim = input_dim
        print(f"Модель загружена из {self.model_path}")

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет препроцессинг к сырым данным
        Args:
            df: DataFrame с сырыми признаками
        Returns:
            DataFrame с обработанными и масштабированными признаками,
            содержащий только те колонки, которые используются моделью
        """
        preprocessor = DataPreprocessor()
        preprocessor.scaler.set_scaler_params({
            'scaler': self.scaler, 'feature_columns': self.feature_columns})
        preprocessor.encoder.encoder = self.encoder
        preprocessor.encoder.categories = self.categories
        df_eng = preprocessor.feature_engineer.transform(df)
        df_enc = preprocessor.encoder.transform(df_eng)
        df_enc = df_enc[self.full_feature_columns]
        df_scaled = preprocessor.scaler.transform(df_enc)
        return df_scaled

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Выполняет прогнозирование для нового DataFrame
        Args:
            df: DataFrame с сырыми признаками
        Returns:
            DataFrame с колонками:
                - 'T1': вероятность инцидента (0..1)
                - 'T2': логарифм условного ущерба (только для инцидентов,
                    NaN для остальных)
                - 'T3': логарифм безусловного ущерба (для всех компаний)
        """
        df_processed = self.preprocess_data(df)
        X = df_processed.values.astype(np.float32)
        X_tensor = torch.tensor(X, device=self.device)
        if self.model is None:
            input_dim = X.shape[1]
            self.load_model(input_dim)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            t1 = torch.sigmoid(outputs['t1']).cpu().numpy().flatten()
            t2 = outputs['t2'].cpu().numpy().flatten()
            t3 = outputs['t3'].cpu().numpy().flatten()
        return pd.DataFrame({'T1': t1, 'T2': t2, 'T3': t3})
