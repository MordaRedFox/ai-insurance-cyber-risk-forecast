"""
Модульные тесты для пакета инференса (src/infer)

Проверяют:
- Загрузку препроцессора (scaler, encoder, списки признаков)
- Преобразование сырых данных через preprocess_data
- Прогнозирование через predict (размерность, типы, допустимые значения)
- Загрузку модели и корректность форвардного прохода
"""

import pytest
import torch
import numpy as np
import sys
sys.path.append('.')
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import DataLoader, TensorDataset
from src.infer import Inference
from src.model import CyberRiskModel
from src.preprocessing import DataPreprocessor
from src.data_generation import DataGenerator
from src.utils import load_config, set_seed, load_dataframe, save_config


SEED: int = 42
TEST_SIZE: int = 100
BATCH_SIZE: int = 8
NUM_EPOCHS: int = 2
HIDDEN_DIMS: list = [8, 4]
DROPOUT: float = 0.1
HEAD_DROPOUT: float = 0.1
LEARNING_RATE: float = 0.01
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope="module")
def config() -> Dict[str, Any]:
    """
    Загружает конфигурационный файл генерации
    Returns:
        Словарь с параметрами конфигурации
    """
    config_path = Path('configs/generation_config.yaml')
    return load_config(config_path)

@pytest.fixture(scope="module")
def temp_data_and_model(tmp_path_factory) -> Dict[str, Path]:
    """
    Генерирует временные данные, обучает минимальную модель и сохраняет
    препроцессор и модель для использования в тестах
    Returns:
        Словарь с путями:
            - 'raw_data': путь к CSV с сырыми данными
            - 'preprocessor': путь к сохраненному препроцессору
            - 'model': путь к сохраненной модели
            - 'model_config': путь к конфигу модели (временный)
    """
    set_seed(SEED)
    tmp_dir = tmp_path_factory.mktemp('inference_test')
    # Генерация сырых данных
    config_path = Path('configs/generation_config.yaml')
    gen = DataGenerator(config_path)
    gen.num_samples = TEST_SIZE
    df_raw = gen.generate_dataset()
    raw_path = tmp_dir / 'raw_companies.csv'
    df_raw.to_csv(raw_path, index=False)
    # Препроцессинг
    preprocessor = DataPreprocessor()
    processed_path = tmp_dir / 'processed_companies.csv'
    scaler_path = tmp_dir / 'scaler.pkl'
    df_processed = preprocessor.process(
        input_path=raw_path, output_path=processed_path,
        save_scaler=True, scaler_path=scaler_path)
    # Подготовка данных для обучения
    target_cols = ['T1', 'T2', 'T3']
    exclude_cols = target_cols + [
        'Фактический совокупный ущерб за последние 12 месяцев (млн руб.)']
    feature_cols = [
        col for col in df_processed.columns if col not in exclude_cols]
    X = df_processed[feature_cols].values.astype(np.float32)
    y = df_processed[target_cols].values.astype(np.float32)
    np.random.seed(SEED)
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * 0.8)
    train_idx, val_idx = indices[split_idx:], indices[:split_idx]
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)
    train_dataset = TensorDataset(
        X_train_t, y_train_t[:, 0], y_train_t[:, 1], y_train_t[:, 2])
    val_dataset = TensorDataset(
        X_val_t, y_val_t[:, 0], y_val_t[:, 1], y_val_t[:, 2])
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # Создание и обучение минимальной модели
    input_dim = len(feature_cols)
    model = CyberRiskModel(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
        head_dropout=HEAD_DROPOUT,
        activation='relu',
        use_batch_norm=False
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion_t1 = torch.nn.BCEWithLogitsLoss()
    criterion_t2 = torch.nn.MSELoss()
    criterion_t3 = torch.nn.MSELoss()
    model.train()
    for epoch in range(NUM_EPOCHS):
        for batch in train_loader:
            Xb, y1, y2, y3 = batch
            optimizer.zero_grad()
            outputs = model(Xb)
            t1_logits = outputs['t1']
            t2_pred = outputs['t2']
            t3_pred = outputs['t3']
            loss_t1 = criterion_t1(t1_logits, y1.unsqueeze(1))
            mask_t2 = (y1 > 0.5).squeeze()
            if mask_t2.sum() > 0:
                loss_t2 = criterion_t2(
                    t2_pred[mask_t2], y2[mask_t2].unsqueeze(1))
            else:
                loss_t2 = torch.tensor(0.0)
            loss_t3 = criterion_t3(t3_pred, y3.unsqueeze(1))
            loss = loss_t1 + loss_t2 + loss_t3
            loss.backward()
            optimizer.step()
    model_path = tmp_dir / 'best_model.pth'
    torch.save(model.state_dict(), model_path)
    # Создание временного конфига модели
    model_config_dict = {
        'model': {
            'input_dim': None,
            'hidden_dims': HIDDEN_DIMS,
            'dropout': DROPOUT,
            'activation': 'relu',
            'use_batch_norm': False,
            'head_dropout': HEAD_DROPOUT
        },
        'loss_weights': {'t1': 1.0, 't2': 1.0, 't3': 1.0}
    }
    model_config_path = tmp_dir / 'model_config.yaml'
    save_config(model_config_dict, model_config_path)
    return {'raw_data': raw_path, 'preprocessor': scaler_path,
            'model': model_path, 'model_config': model_config_path,}


class TestInference:
    """
    Тесты для класса Inference
    """

    def test_load_preprocessor(
            self, temp_data_and_model: Dict[str, Path]) -> None:
        """
        Проверяет, что препроцессор загружается корректно (scaler, encoder,
        feature_columns, categories, full_feature_columns)
        Args:
            temp_data_and_model: словарь с путями к временным данным и модели
        """
        inf = Inference(
            model_path=temp_data_and_model['model'],
            preprocessor_path=temp_data_and_model['preprocessor'],
            model_config=temp_data_and_model['model_config'],
            device='cpu'
        )
        assert inf.scaler is not None
        assert inf.feature_columns is not None
        assert len(inf.feature_columns) > 0
        assert inf.encoder is not None
        assert inf.categories is not None
        assert inf.full_feature_columns is not None
        assert len(inf.full_feature_columns) > 0
        assert set(inf.feature_columns).issubset(set(inf.full_feature_columns))

    def test_preprocess_data(
            self, temp_data_and_model: Dict[str, Path]) -> None:
        """
        Проверяет, что preprocess_data возвращает DataFrame с правильными
        колонками и без NaN/inf
        Args:
            temp_data_and_model: словарь с путями к временным данным и модели
        """
        inf = Inference(
            model_path=temp_data_and_model['model'],
            preprocessor_path=temp_data_and_model['preprocessor'],
            model_config=temp_data_and_model['model_config'],
            device='cpu'
        )
        df_raw = load_dataframe(temp_data_and_model['raw_data']).head(10)
        df_processed = inf.preprocess_data(df_raw)
        assert set(df_processed.columns) == set(inf.full_feature_columns)
        assert not df_processed.isnull().any().any()
        assert not np.isinf(df_processed.values).any()

    def test_predict_shape_and_types(
            self, temp_data_and_model: Dict[str, Path]) -> None:
        """
        Проверяет, что predict возвращает DataFrame с нужными колонками,
        правильной размерностью и допустимыми значениями
        Args:
            temp_data_and_model: словарь с путями к временным данным и модели
        """
        inf = Inference(
            model_path=temp_data_and_model['model'],
            preprocessor_path=temp_data_and_model['preprocessor'],
            model_config=temp_data_and_model['model_config'],
            device='cpu'
        )
        df_raw = load_dataframe(temp_data_and_model['raw_data']).head(10)
        predictions = inf.predict(df_raw)
        expected_cols = {'T1', 'T2', 'T3'}
        assert set(predictions.columns) == expected_cols
        assert len(predictions) == len(df_raw)
        assert (predictions['T1'] >= 0).all() and (predictions['T1'] <= 1).all()
        assert np.isfinite(predictions['T2']).all()
        assert np.isfinite(predictions['T3']).all()

    def test_model_loading_and_forward(
            self, temp_data_and_model: Dict[str, Path]) -> None:
        """
        Проверяет, что модель загружается и выполняет прямой проход без ошибок
        Args:
            temp_data_and_model: словарь с путями к временным данным и модели
        """
        inf = Inference(
            model_path=temp_data_and_model['model'],
            preprocessor_path=temp_data_and_model['preprocessor'],
            model_config=temp_data_and_model['model_config'],
            device='cpu'
        )
        df_raw = load_dataframe(temp_data_and_model['raw_data']).head(10)
        df_processed = inf.preprocess_data(df_raw)
        X = df_processed.values.astype(np.float32)
        X_tensor = torch.tensor(X, device=inf.device)
        input_dim = X.shape[1]
        inf.load_model(input_dim)
        with torch.no_grad():
            outputs = inf.model(X_tensor)
        assert 't1' in outputs
        assert 't2' in outputs
        assert 't3' in outputs
        assert outputs['t1'].shape == (X.shape[0], 1)
        assert outputs['t2'].shape == (X.shape[0], 1)
        assert outputs['t3'].shape == (X.shape[0], 1)

    def test_predict_without_preprocessing(
            self, temp_data_and_model: Dict[str, Path]) -> None:
        """
        Проверяет, что predict автоматически вызывает препроцессинг и
        загрузку модели
        Args:
            temp_data_and_model: словарь с путями к временным данным и модели
        """
        inf = Inference(
            model_path=temp_data_and_model['model'],
            preprocessor_path=temp_data_and_model['preprocessor'],
            model_config=temp_data_and_model['model_config'],
            device='cpu'
        )
        df_raw = load_dataframe(temp_data_and_model['raw_data']).head(5)
        assert inf.model is None
        predictions = inf.predict(df_raw)
        assert inf.model is not None
        assert len(predictions) == len(df_raw)
        assert (predictions['T1'] >= 0).all() and (predictions['T1'] <= 1).all()
