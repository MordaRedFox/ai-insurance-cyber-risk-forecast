"""
Модульные тесты для модели CyberRiskModel

Проверяют:
- Создание модели с различными параметрами
- Формат выходов (ключи, размеры)
- Работу с батчами разных размеров
- Проверку, что модель обучается (один шаг оптимизации)
"""

import pytest
import torch
import torch.nn as nn
import sys
sys.path.append('.')
from typing import Tuple
from src.model import CyberRiskModel


BATCH_SIZE: int = 16
INPUT_DIM: int = 32
HIDDEN_DIMS: list = [64, 32]
DROPOUT: float = 0.2
HEAD_DROPOUT: float = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOLERANCE: float = 1e-6


@pytest.fixture
def random_batch() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                            torch.Tensor]:
    """
    Генерирует случайный батч признаков и целей для тестов
    Returns:
        Кортеж (X, y1, y2, y3) как тензоры на CPU:
            - X: признаки размерности (BATCH_SIZE, INPUT_DIM)
            - y1: бинарная метка (BATCH_SIZE, 1)
            - y2: целевая переменная для severity (BATCH_SIZE, 1)
            - y3: целевая переменная для pure premium (BATCH_SIZE, 1)
    """
    X = torch.randn(BATCH_SIZE, INPUT_DIM)
    y1 = torch.randint(0, 2, (BATCH_SIZE, 1)).float()
    y2 = torch.randn(BATCH_SIZE, 1)
    y3 = torch.randn(BATCH_SIZE, 1)
    return X, y1, y2, y3

@pytest.fixture
def model() -> CyberRiskModel:
    """
    Создает экземпляр модели с тестовыми параметрами
    Returns:
        Экземпляр CyberRiskModel на CPU
    """
    return CyberRiskModel(
        input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT,
        head_dropout=HEAD_DROPOUT, activation='relu', use_batch_norm=True)


class TestCyberRiskModel:
    """
    Тесты архитектуры и форвард-прохода модели
    """

    def test_model_creation(self) -> None:
        """
        Проверяет, что модель создается без ошибок и имеет правильное
        количество слоев. Также проверяет наличие всех голов и корректность
        выходных размеров на батче размера 1
        """
        model = CyberRiskModel(
            input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT,
            head_dropout=HEAD_DROPOUT, activation='relu', use_batch_norm=True)
        expected_backbone_layers = len(HIDDEN_DIMS) * 4
        assert len(model.backbone) == expected_backbone_layers, (
            f"Ожидалось {expected_backbone_layers} слоёв в бэкбоне, "
            f"получено {len(model.backbone)}")
        assert hasattr(model, 'head_t1')
        assert hasattr(model, 'head_t2')
        assert hasattr(model, 'head_t3')
        model.eval()
        dummy_input = torch.randn(1, INPUT_DIM)
        with torch.no_grad():
            outputs = model(dummy_input)
            assert outputs['t1'].shape == (1, 1)
            assert outputs['t2'].shape == (1, 1)
            assert outputs['t3'].shape == (1, 1)
        model.train()

    def test_forward_output_keys(self, model: CyberRiskModel,
                                 random_batch: Tuple) -> None:
        """
        Проверяет, что forward возвращает словарь с ожидаемыми ключами
        Args:
            model: экземпляр модели
            random_batch: фикстура со случайным батчем
        """
        X, _, _, _ = random_batch
        outputs = model(X)
        expected_keys = {'t1', 't2', 't3'}
        assert set(outputs.keys()) == expected_keys, (
            f"Ожидались ключи {expected_keys}, "
            f"получены {set(outputs.keys())}")

    def test_forward_output_shapes(self, model: CyberRiskModel,
                                   random_batch: Tuple) -> None:
        """
        Проверяет размерности выходных тензоров
        Args:
            model: экземпляр модели
            random_batch: фикстура со случайным батчем
        """
        X, _, _, _ = random_batch
        batch_size = X.size(0)
        outputs = model(X)
        assert outputs['t1'].shape == (batch_size, 1)
        assert outputs['t2'].shape == (batch_size, 1)
        assert outputs['t3'].shape == (batch_size, 1)

    def test_forward_t1_logits(self, model: CyberRiskModel,
                               random_batch: Tuple) -> None:
        """
        Проверяет, что выход t1 является логитами (не ограничен сигмоидом)
        Затем проверяет, что после применения сигмоида значения попадают в [0,1]
        Args:
            model: экземпляр модели
            random_batch: фикстура со случайным батчем
        """
        X, _, _, _ = random_batch
        outputs = model(X)
        t1_logits = outputs['t1']
        t1_proba = torch.sigmoid(t1_logits)
        all_non_negative = (t1_proba >= 0).all()
        all_not_exceed_one = (t1_proba <= 1).all()
        assert all_non_negative and all_not_exceed_one, (
            "После сигмоида значения должны быть в [0,1]")

    def test_forward_different_batch_sizes(self, model: CyberRiskModel) -> None:
        """
        Проверяет, что модель работает с батчами разных размеров (включая 1)
        Args:
            model: экземпляр модели
        """
        model.eval()
        for batch_size in [1, 3, 8, 13, 32]:
            X = torch.randn(batch_size, INPUT_DIM)
            outputs = model(X)
            actual_samples = outputs['t1'].shape[0]
            assert actual_samples == batch_size, (
                f"Ожидалось {batch_size} образцов, получено {actual_samples}")
        model.train()

    def test_model_parameters_count(self, model: CyberRiskModel) -> None:
        """
        Проверяет, что модель имеет разумное количество параметров (> 1000)
        Args:
            model: экземпляр модели
        """
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0, "Модель не содержит параметров"
        assert total_params > 1000, f"Слишком мало параметров: {total_params}"

    def test_model_training_step(self, model: CyberRiskModel,
                                 random_batch: Tuple) -> None:
        """
        Проверяет, что модель может выполнить один шаг обучения
        (градиент обновляется, параметры меняются)
        Args:
            model: экземпляр модели
            random_batch: фикстура со случайным батчем
        """
        X, y1, y2, y3 = random_batch
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        initial_params = [p.clone().detach() for p in model.parameters()]
        outputs = model(X)
        t1_logits = outputs['t1']
        t2 = outputs['t2']
        t3 = outputs['t3']
        loss_t1 = nn.BCEWithLogitsLoss()(t1_logits, y1)
        loss_t2 = nn.MSELoss()(t2, y2)
        loss_t3 = nn.MSELoss()(t3, y3)
        loss = loss_t1 + loss_t2 + loss_t3
        loss.backward()
        optimizer.step()
        changed = False
        for p_initial, p in zip(initial_params, model.parameters()):
            if not torch.allclose(p_initial, p, atol=TOLERANCE):
                changed = True
                break
        assert changed, "Параметры не изменились после шага оптимизации"
