"""
Модульные тесты для пакета data_generation

Проверяют корректность генерации синтетических данных:
- Размерности
- Отсутствие NaN и inf
- Соответствие распределений заданным вероятностям
- Долю нулевых значений в целевой переменной
"""

import pytest
import numpy as np
import pandas as pd
import sys
sys.path.append('.')
from pathlib import Path
from typing import Dict, Any
from src.data_generation import (
    DataGenerator, IndustryGenerator, ContinuousFeatureGenerator,
    ShareFeatureGenerator, BinaryFeatureGenerator, PentestGenerator,
    SecuritySpendGenerator, TargetGenerator
)
from src.utils import load_config, set_seed


SEED: int = 42
DEFAULT_NUM_SAMPLES: int = 100000
ALLOWED_DEVIATION: float = 0.05
CLIP_MAX: float = 1e9


@pytest.fixture(scope="module")
def config() -> Dict[str, Any]:
    """
    Загружает конфигурационный файл генерации
    Returns:
        Словарь с параметрами конфигурации
    """
    config_path = Path("configs/generation_config.yaml")
    return load_config(config_path)

@pytest.fixture(scope="module")
def rng() -> np.random.RandomState:
    """
    Возвращает генератор случайных чисел с фиксированным seed
    Returns:
        Объект RandomState
    """
    set_seed(SEED)
    return np.random.RandomState(SEED)

@pytest.fixture(scope="module")
def num_samples() -> int:
    """
    Возвращает количество образцов для тестов
    Returns:
        Количество образцов
    """
    return DEFAULT_NUM_SAMPLES

@pytest.fixture(scope="module")
def industries(config: Dict[str, Any], rng: np.random.RandomState,
               num_samples: int) -> np.ndarray:
    """
    Генерирует массив отраслей для тестов
    Args:
        config: конфигурация
        rng: генератор случайных чисел
        num_samples: количество образцов
    Returns:
        Массив отраслей размером (num_samples,)
    """
    gen = IndustryGenerator(config, rng)
    gen.num_samples = num_samples
    return gen.generate()

@pytest.fixture(scope="module")
def feature_dataframe(config: Dict[str, Any], rng: np.random.RandomState,
                      num_samples: int, industries: np.ndarray) -> pd.DataFrame:
    """
    Создает полный DataFrame со всеми признаками (без целевой переменной)
    Args:
        config: конфигурация
        rng: генератор случайных чисел
        num_samples: количество образцов
        industries: массив отраслей
    Returns:
        DataFrame с признаками
    """
    # Непрерывные признаки
    cont_gen = ContinuousFeatureGenerator(config, rng)
    cont_gen.num_samples = num_samples
    df_cont = cont_gen.generate()
    # Расходы на безопасность
    sec_gen = SecuritySpendGenerator(config, rng)
    sec_gen.num_samples = num_samples
    security_spend = sec_gen.generate(df_cont['revenue'].values, industries)
    df_cont['security_spend'] = security_spend
    # Долевые признаки
    share_gen = ShareFeatureGenerator(config, rng)
    share_gen.num_samples = num_samples
    df_shares = share_gen.generate(industries)
    # Бинарные признаки
    bin_gen = BinaryFeatureGenerator(config, rng)
    bin_gen.num_samples = num_samples
    df_binary = bin_gen.generate(industries)
    # Периодичность пентеста
    pentest_gen = PentestGenerator(config, rng)
    pentest_gen.num_samples = num_samples
    pentest = pentest_gen.generate(industries)
    # Признаки простоя
    log_params = config['log_normal_params']
    names = ['downtime_hour_loss', 'mtd_hours']
    means = [log_params['downtime_hour_loss']['mean'],
             log_params['mtd_hours']['mean']]
    stds = [log_params['downtime_hour_loss']['std'],
            log_params['mtd_hours']['std']]
    log_vals = rng.normal(loc=means, scale=stds, size=(num_samples, 2))
    vals = np.exp(log_vals)
    vals = np.clip(vals, 0, CLIP_MAX)
    df_downtime = pd.DataFrame(vals, columns=names)
    # Сборка итогового DataFrame
    df = pd.concat([df_cont, df_shares, df_binary,
                    pd.DataFrame({'pentest': pentest}), df_downtime], axis=1)
    df['industry'] = industries
    return df


class TestIndustryGenerator:
    """
    Тесты для генератора отраслей
    """

    def test_generate_shape(self, config: Dict[str, Any],
                            rng: np.random.RandomState,
                            num_samples: int) -> None:
        """
        Проверяет, что генератор возвращает массив правильной размерности
        Args:
            config: конфигурация
            rng: генератор случайных чисел
            num_samples: ожидаемое количество образцов
        """
        gen = IndustryGenerator(config, rng)
        gen.num_samples = num_samples
        industries = gen.generate()
        assert len(industries) == num_samples
        assert isinstance(industries, np.ndarray)
        assert industries.dtype.kind == 'U' or industries.dtype == object

    def test_industry_values(self, config: Dict[str, Any],
                             rng: np.random.RandomState,
                             num_samples: int) -> None:
        """
        Проверяет, что все значения отраслей допустимы
        Args:
            config: конфигурация
            rng: генератор случайных чисел
            num_samples: количество образцов
        """
        gen = IndustryGenerator(config, rng)
        gen.num_samples = num_samples
        industries = gen.generate()
        valid = set(gen.industry_names)
        assert set(industries) <= valid


class TestContinuousFeatureGenerator:
    """
    Тесты для генератора непрерывных признаков
    """

    def test_generate_shape(self, config: Dict[str, Any],
                            rng: np.random.RandomState,
                            num_samples: int) -> None:
        """
        Проверяет размерность DataFrame
        Args:
            config: конфигурация
            rng: генератор случайных чисел
            num_samples: количество образцов
        """
        gen = ContinuousFeatureGenerator(config, rng)
        gen.num_samples = num_samples
        df = gen.generate()
        assert df.shape == (num_samples, len(gen.FEATURE_NAMES))
        assert all(col in df.columns for col in gen.FEATURE_NAMES)

    def test_no_nan_inf(self, config: Dict[str, Any],
                        rng: np.random.RandomState,
                        num_samples: int) -> None:
        """
        Проверяет отсутствие NaN и inf в сгенерированных данных
        Args:
            config: конфигурация
            rng: генератор случайных чисел
            num_samples: количество образцов
        """
        gen = ContinuousFeatureGenerator(config, rng)
        gen.num_samples = num_samples
        df = gen.generate()
        assert not df.isnull().any().any()
        assert not np.isinf(df.values).any()

    def test_zero_masking(self, config: Dict[str, Any],
                          rng: np.random.RandomState,
                          num_samples: int) -> None:
        """
        Проверяет, что маскирование нулей работает с заданной вероятностью
        Args:
            config: конфигурация
            rng: генератор случайных чисел
            num_samples: количество образцов
        """
        gen = ContinuousFeatureGenerator(config, rng)
        gen.num_samples = num_samples
        df = gen.generate()
        zero_probs = config.get('zero_probabilities', {})
        for col, prob in zero_probs.items():
            if col in df.columns:
                zero_frac = (df[col] == 0).mean()
                assert abs(zero_frac - prob) < ALLOWED_DEVIATION


class TestShareFeatureGenerator:
    """
    Тесты для генератора долевых признаков
    """

    def test_generate_shape(self, config: Dict[str, Any],
                            rng: np.random.RandomState, num_samples: int,
                            industries: np.ndarray) -> None:
        """
        Проверяет размерность DataFrame
        Args:
            config: конфигурация
            rng: генератор случайных чисел
            num_samples: количество образцов
            industries: массив отраслей
        """
        gen = ShareFeatureGenerator(config, rng)
        gen.num_samples = num_samples
        df = gen.generate(industries)
        assert df.shape == (num_samples, len(gen.SHARE_NAMES))
        assert all(col in df.columns for col in gen.SHARE_NAMES)

    def test_values_in_range(self, config: Dict[str, Any],
                             rng: np.random.RandomState, num_samples: int,
                             industries: np.ndarray) -> None:
        """
        Проверяет, что все значения лежат в диапазоне [0, 1]
        Args:
            config: конфигурация
            rng: генератор случайных чисел
            num_samples: количество образцов
            industries: массив отраслей
        """
        gen = ShareFeatureGenerator(config, rng)
        gen.num_samples = num_samples
        df = gen.generate(industries)
        for col in gen.SHARE_NAMES:
            assert (df[col] >= 0).all()
            assert (df[col] <= 1).all()


class TestBinaryFeatureGenerator:
    """
    Тесты для генератора бинарных признаков
    """

    def test_generate_shape(self, config: Dict[str, Any],
                            rng: np.random.RandomState, num_samples: int,
                            industries: np.ndarray) -> None:
        """
        Проверяет размерность DataFrame
        Args:
            config: конфигурация
            rng: генератор случайных чисел
            num_samples: количество образцов
            industries: массив отраслей
        """
        gen = BinaryFeatureGenerator(config, rng)
        gen.num_samples = num_samples
        df = gen.generate(industries)
        assert df.shape == (num_samples, len(gen.BINARY_NAMES))
        assert all(col in df.columns for col in gen.BINARY_NAMES)

    def test_values_binary(self, config: Dict[str, Any],
                           rng: np.random.RandomState, num_samples: int,
                           industries: np.ndarray) -> None:
        """
        Проверяет, что все значения равны 0 или 1
        Args:
            config: конфигурация
            rng: генератор случайных чисел
            num_samples: количество образцов
            industries: массив отраслей
        """
        gen = BinaryFeatureGenerator(config, rng)
        gen.num_samples = num_samples
        df = gen.generate(industries)
        for col in gen.BINARY_NAMES:
            assert set(df[col].unique()) <= {0, 1}


class TestPentestGenerator:
    """
    Тесты для генератора периодичности пентеста
    """

    def test_generate_shape(self, config: Dict[str, Any],
                            rng: np.random.RandomState, num_samples: int,
                            industries: np.ndarray) -> None:
        """
        Проверяет размерность и допустимые значения
        Args:
            config: конфигурация
            rng: генератор случайных чисел
            num_samples: количество образцов
            industries: массив отраслей
        """
        gen = PentestGenerator(config, rng)
        gen.num_samples = num_samples
        pentest = gen.generate(industries)
        assert len(pentest) == num_samples
        assert set(pentest) <= set(gen.CATEGORIES)

    def test_probability_distribution(self, config: Dict[str, Any],
                                      rng: np.random.RandomState,
                                      num_samples: int) -> None:
        """
        Проверяет, что распределение пентеста соответствует
        заданным вероятностям
        Args:
            config: конфигурация
            rng: генератор случайных чисел
            num_samples: количество образцов
        """
        industry = list(config['industry_probs'].keys())[0]
        industries = np.array([industry] * num_samples)
        gen = PentestGenerator(config, rng)
        gen.num_samples = num_samples
        pentest = gen.generate(industries)
        probs = config['pentest_probs'][industry]
        for cat in gen.CATEGORIES:
            freq = (pentest == cat).mean()
            expected = probs[cat]
            assert abs(freq - expected) < ALLOWED_DEVIATION


class TestSecuritySpendGenerator:
    """
    Тесты для генератора расходов на безопасность
    """

    def test_generate_shape(self, config: Dict[str, Any],
                            rng: np.random.RandomState, num_samples: int,
                            industries: np.ndarray) -> None:
        """
        Проверяет размерность и отсутствие отрицательных значений
        Args:
            config: конфигурация
            rng: генератор случайных чисел
            num_samples: количество образцов
            industries: массив отраслей
        """
        cont_gen = ContinuousFeatureGenerator(config, rng)
        cont_gen.num_samples = num_samples
        df_cont = cont_gen.generate()
        revenue = df_cont['revenue'].values
        gen = SecuritySpendGenerator(config, rng)
        gen.num_samples = num_samples
        spend = gen.generate(revenue, industries)
        assert len(spend) == num_samples
        assert (spend >= 0).all()
        assert not np.isinf(spend).any()

    def test_security_spend_nonzero(self, config: Dict[str, Any],
                                    rng: np.random.RandomState,
                                    num_samples: int,
                                    industries: np.ndarray) -> None:
        """
        Проверяет, что есть хотя бы одно положительное значение
        Args:
            config: конфигурация
            rng: генератор случайных чисел
            num_samples: количество образцов
            industries: массив отраслей
        """
        cont_gen = ContinuousFeatureGenerator(config, rng)
        cont_gen.num_samples = num_samples
        df_cont = cont_gen.generate()
        revenue = df_cont['revenue'].values
        gen = SecuritySpendGenerator(config, rng)
        gen.num_samples = num_samples
        spend = gen.generate(revenue, industries)
        assert (spend > 0).any()


class TestTargetGenerator:
    """
    Тесты для генератора целевой переменной Y0
    """

    def test_generate_no_nan_inf(self, config: Dict[str, Any],
                                  rng: np.random.RandomState, num_samples: int,
                                  feature_dataframe: pd.DataFrame) -> None:
        """
        Проверяет, что целевая переменная не содержит NaN и inf
        Args:
            config: конфигурация
            rng: генератор случайных чисел
            num_samples: количество образцов
            feature_dataframe: DataFrame с признаками
        """
        gen = TargetGenerator(config, rng)
        gen.num_samples = num_samples
        y = gen.generate(feature_dataframe)
        assert len(y) == num_samples
        assert not np.isnan(y).any()
        assert not np.isinf(y).any()
        assert (y > 0).any()

    def test_zero_ratio(self, config: Dict[str, Any],
                        rng: np.random.RandomState, num_samples: int,
                        feature_dataframe: pd.DataFrame) -> None:
        """
        Проверяет, что доля нулевых значений близка к заданной
        Args:
            config: конфигурация
            rng: генератор случайных чисел
            num_samples: количество образцов
            feature_dataframe: DataFrame с признаками
        """
        gen = TargetGenerator(config, rng)
        gen.num_samples = num_samples
        y = gen.generate(feature_dataframe)
        zero_frac = (y == 0).mean()
        expected = config['zero_loss_prob']
        assert abs(zero_frac - expected) < ALLOWED_DEVIATION


class TestIntegration:
    """
    Интеграционный тест полного пайплайна генерации
    """

    def test_full_generation_no_nan_inf(self, tmp_path: Path,
                                        config: Dict[str, Any],
                                        num_samples: int) -> None:
        """
        Проверяет, что полный процесс генерации не дает NaN и inf
        Args:
            tmp_path: временная директория для конфигурационного файла
            config: конфигурация
            num_samples: количество образцов
        """
        import yaml
        temp_config_path = tmp_path / "test_config.yaml"
        config['num_samples'] = num_samples
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)

        gen = DataGenerator(temp_config_path)
        df = gen.generate_dataset()
        target_col = 'Фактический совокупный ущерб за последние 12 месяцев (млн руб.)'
        y = df[target_col].values
        assert not np.isnan(y).any()
        assert not np.isinf(y).any()
        assert (y > 0).any()
