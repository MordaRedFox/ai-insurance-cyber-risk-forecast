"""
Модульные тесты для пакета обучения (src/train)

Проверяют:
- Корректность функции потерь pinball_loss
- Подготовку DataLoader'ов (размеры, типы)
- Инициализацию Trainer и выполнение одного шага обучения
- Методы evaluate и сохранение модели
"""

import pytest
import torch
import sys
sys.path.append('.')
import numpy as np
import pandas as pd
from pathlib import Path
from src.train import Trainer, prepare_dataloaders, pinball_loss
from src.model import CyberRiskModel


BATCH_SIZE: int = 8
INPUT_DIM: int = 10
NUM_SAMPLES: int = 100
FEATURE_COLS = [f'feature_{i}' for i in range(INPUT_DIM)]
TARGET_COLS = ['T1', 'T2', 'T3', 'T4']
SEED: int = 42
TOLERANCE: float = 1e-6


@pytest.fixture(scope="module")
def random_dataframe() -> pd.DataFrame:
    """
    Создает случайный DataFrame с признаками и целями
    Returns:
        DataFrame с колонками feature_* и T1..T4
    """
    np.random.seed(SEED)
    X = np.random.randn(NUM_SAMPLES, INPUT_DIM)
    y1 = np.random.randint(0, 2, (NUM_SAMPLES, 1)).astype(np.float32)
    y2 = np.random.randn(NUM_SAMPLES, 1).astype(np.float32)
    y3 = np.random.randn(NUM_SAMPLES, 1).astype(np.float32)
    y4 = np.random.randn(NUM_SAMPLES, 1).astype(np.float32)
    data = {f'feature_{i}': X[:, i] for i in range(INPUT_DIM)}
    data['T1'] = y1.flatten()
    data['T2'] = y2.flatten()
    data['T3'] = y3.flatten()
    data['T4'] = y4.flatten()
    return pd.DataFrame(data)

@pytest.fixture
def model() -> CyberRiskModel:
    """
    Создает небольшую модель для тестов
    Returns:
        Модель искусственного интеллекта
    """
    return CyberRiskModel(
        input_dim=INPUT_DIM, hidden_dims=[8, 4], dropout=0.1, head_dropout=0.1,
        activation='relu', use_batch_norm=False)

@pytest.fixture
def train_config() -> dict:
    """
    Базовая конфигурация обучения
    Returns:
        Словарь с конфигурацией для обучения
    """
    return {
        'learning_rate': 0.01,
        'weight_decay': 1e-5,
        'scheduler_patience': 2,
        'scheduler_factor': 0.5,
        'early_stopping_patience': 3,
        'loss_weights': {'t1': 1.0, 't2': 1.0, 't3': 1.0, 't4': 1.0}
    }


class TestPinballLoss:
    """
    Тесты для функции pinball_loss
    """

    def test_shape(self) -> None:
        """
        Проверяет, что функция возвращает скаляр
        """
        pred = torch.randn(10, 1)
        target = torch.randn(10, 1)
        loss = pinball_loss(pred, target, tau=0.9)
        assert loss.ndim == 0

    def test_zero_error(self) -> None:
        """
        Проверяет, что при нулевой ошибке потеря равна 0
        """
        target = torch.ones(5, 1)
        pred = target.clone()
        loss = pinball_loss(pred, target, tau=0.9)
        assert torch.isclose(
            loss, torch.tensor(0.0, dtype=torch.float32), atol=TOLERANCE)

    def test_positive_error(self) -> None:
        """
        Проверяет, что при ошибке (занижение/завышение) потеря > 0
        """
        target = torch.ones(5, 1)
        pred_low = torch.zeros(5, 1)
        loss_low = pinball_loss(pred_low, target, tau=0.9)
        assert loss_low > 0
        target_zero = torch.zeros(5, 1)
        pred_high = torch.ones(5, 1)
        loss_high = pinball_loss(pred_high, target_zero, tau=0.9)
        assert loss_high > 0

    def test_asymmetry(self) -> None:
        """
        Проверяет асимметрию: при tau=0.9 занижение наказывается сильнее
        """
        target = torch.zeros(1, 1)
        pred_low = torch.tensor([[-1.0]])
        pred_high = torch.tensor([[1.0]])
        loss_low = pinball_loss(pred_low, target, tau=0.9)
        loss_high = pinball_loss(pred_high, target, tau=0.9)
        assert loss_low > loss_high

    def test_tau_range(self) -> None:
        """
        Проверяет, что функция работает для разных tau (0.1, 0.5, 0.9)
        """
        pred = torch.randn(10, 1)
        target = torch.randn(10, 1)
        for tau in [0.1, 0.5, 0.9]:
            loss = pinball_loss(pred, target, tau=tau)
            assert not torch.isnan(loss)


class TestPrepareDataloaders:
    """
    Тесты для prepare_dataloaders
    """

    def test_output_types(self, random_dataframe: pd.DataFrame) -> None:
        """
        Проверяет, что prepare_dataloaders возвращает два объекта DataLoader
        Args:
            random_dataframe: фикстура с тестовыми данными
        """
        train_loader, val_loader = prepare_dataloaders(
            random_dataframe, feature_cols=FEATURE_COLS,
            target_cols=TARGET_COLS, batch_size=BATCH_SIZE,
            val_split=0.2, random_state=SEED)
        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert isinstance(val_loader, torch.utils.data.DataLoader)

    def test_split_sizes(self, random_dataframe: pd.DataFrame) -> None:
        """
        Проверяет, что размеры выборок соответствуют val_split
        Args:
            random_dataframe: фикстура с тестовыми данными
        """
        val_split = 0.2
        train_loader, val_loader = prepare_dataloaders(
            random_dataframe, feature_cols=FEATURE_COLS,
            target_cols=TARGET_COLS, batch_size=BATCH_SIZE,
            val_split=val_split, random_state=SEED)
        total_samples = len(random_dataframe)
        train_samples = sum(len(batch[0]) for batch in train_loader)
        val_samples = sum(len(batch[0]) for batch in val_loader)
        assert train_samples + val_samples == total_samples

    def test_batch_size(self, random_dataframe: pd.DataFrame) -> None:
        """
        Проверяет, что размер батча не превышает заданный
        Args:
            random_dataframe: фикстура с тестовыми данными
        """
        batch_size = 4
        train_loader, _ = prepare_dataloaders(
            random_dataframe, feature_cols=FEATURE_COLS,
            target_cols=TARGET_COLS, batch_size=batch_size,
            val_split=0.2, random_state=SEED)
        for batch in train_loader:
            X, y1, y2, y3, y4 = batch
            assert X.size(0) <= batch_size
            assert y1.size(0) == X.size(0)
            assert y2.size(0) == X.size(0)
            assert y3.size(0) == X.size(0)
            assert y4.size(0) == X.size(0)
            break

    def test_feature_cols_presence(
            self, random_dataframe: pd.DataFrame) -> None:
        """
        Проверяет, что подготовка данных не изменяет исходный DataFrame
        Args:
            random_dataframe: фикстура с тестовыми данными
        """
        original_cols = set(random_dataframe.columns)
        _ = prepare_dataloaders(
            random_dataframe, feature_cols=FEATURE_COLS,
            target_cols=TARGET_COLS, batch_size=BATCH_SIZE,
            val_split=0.2, random_state=SEED)
        assert set(random_dataframe.columns) == original_cols


class TestTrainer:
    """
    Тесты для класса Trainer
    """

    def test_initialization(self, model: CyberRiskModel,
                            random_dataframe: pd.DataFrame,
                            train_config: dict) -> None:
        """
        Проверяет корректность инициализации Trainer
        Args:
            model: тестовая модель
            random_dataframe: тестовые данные
            train_config: конфигурация обучения
        """
        train_loader, val_loader = prepare_dataloaders(
            random_dataframe, feature_cols=FEATURE_COLS,
            target_cols=TARGET_COLS, batch_size=BATCH_SIZE,
            val_split=0.2, random_state=SEED)
        device = torch.device('cpu')
        trainer = Trainer(
            model, train_loader, val_loader, train_config, device)
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_train_epoch_updates_weights(self, model: CyberRiskModel,
                                         random_dataframe: pd.DataFrame,
                                         train_config: dict) -> None:
        """
        Проверяет, что после одной эпохи параметры модели изменяются
        Args:
            model: тестовая модель
            random_dataframe: тестовые данные
            train_config: конфигурация обучения
        """
        train_loader, val_loader = prepare_dataloaders(
            random_dataframe, feature_cols=FEATURE_COLS,
            target_cols=TARGET_COLS, batch_size=BATCH_SIZE,
            val_split=0.2, random_state=SEED)
        device = torch.device('cpu')
        trainer = Trainer(
            model, train_loader, val_loader, train_config, device)
        initial_params = [p.clone().detach() for p in model.parameters()]
        _ = trainer.train_epoch()
        changed = False
        for p_initial, p in zip(initial_params, model.parameters()):
            if not torch.allclose(p_initial, p, atol=TOLERANCE):
                changed = True
                break
        assert changed, "Параметры должны измениться после одной эпохи"

    def test_evaluate_returns_loss(self, model: CyberRiskModel,
                                   random_dataframe: pd.DataFrame,
                                   train_config: dict) -> None:
        """
        Проверяет, что evaluate возвращает словарь с ключом 'total_loss'
        Args:
            model: тестовая модель
            random_dataframe: тестовые данные
            train_config: конфигурация обучения
        """
        train_loader, val_loader = prepare_dataloaders(
            random_dataframe, feature_cols=FEATURE_COLS,
            target_cols=TARGET_COLS, batch_size=BATCH_SIZE,
            val_split=0.2, random_state=SEED)
        device = torch.device('cpu')
        trainer = Trainer(
            model, train_loader, val_loader, train_config, device)
        metrics = trainer.evaluate(val_loader)
        assert 'total_loss' in metrics
        assert isinstance(metrics['total_loss'], float)

    def test_train_saves_best_model(self, model: CyberRiskModel,
                                    random_dataframe: pd.DataFrame,
                                    train_config: dict,
                                    tmp_path: Path) -> None:
        """
        Проверяет, что train сохраняет лучшую модель и загружает ее в конце
        Args:
            model: тестовая модель
            random_dataframe: тестовые данные
            train_config: конфигурация обучения
            tmp_path: временная директория для сохранения модели
        """
        train_loader, val_loader = prepare_dataloaders(
            random_dataframe, feature_cols=FEATURE_COLS,
            target_cols=TARGET_COLS, batch_size=BATCH_SIZE,
            val_split=0.2, random_state=SEED)
        device = torch.device('cpu')
        trainer = Trainer(
            model, train_loader, val_loader, train_config, device)
        save_dir = tmp_path / 'checkpoints'
        model_name = 'test_model.pth'
        _ = trainer.train(
            num_epochs=3, save_dir=save_dir, model_name=model_name)
        saved_path = save_dir / model_name
        assert saved_path.exists()
        state_dict = torch.load(saved_path)
        for key in state_dict:
            assert torch.allclose(trainer.model.state_dict()[key],
                                  state_dict[key], atol=TOLERANCE)
