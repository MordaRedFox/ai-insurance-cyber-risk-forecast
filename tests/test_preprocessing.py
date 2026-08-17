"""
Модульные тесты для пакета preprocessing

Проверяют корректность препроцессинга данных:
- Загрузка и сохранение данных
- Добавление инженерных признаков (it_density, sec_effectiveness, downtime_cap)
- One-Hot кодирование отрасли
- Масштабирование числовых признаков
- Создание целевых переменных T1-T4
- Сохранение и загрузка scaler
"""

import pytest
import numpy as np
import pandas as pd
import sys
sys.path.append('.')
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.preprocessing import DataPreprocessor
from src.utils import load_dataframe, load_pickle, save_pickle
from src.data_generation import DataGenerator


SEED: int = 42
TEST_SIZE: int = 100000
TOLERANCE: float = 1e-6
MEAN_TOLERANCE: float = 0.1
STD_TOLERANCE: float = 0.1


@pytest.fixture(scope="module")
def raw_data_path(tmp_path_factory) -> Path:
    """
    Генерирует небольшой сырой датасет и сохраняет его во временный файл
    Returns:
        Путь к временному CSV-файлу с сырыми данными
    """
    tmp_dir = tmp_path_factory.mktemp('raw_data')
    config_path = Path('configs/generation_config.yaml')
    gen = DataGenerator(config_path)
    gen.num_samples = TEST_SIZE
    df_raw = gen.generate_dataset()
    file_path = tmp_dir / 'raw_companies.csv'
    df_raw.to_csv(file_path, index=False)
    return file_path

@pytest.fixture(scope="module")
def preprocessor() -> DataPreprocessor:
    """
    Создает экземпляр DataPreprocessor
    Returns:
        Настроенный препроцессор
    """
    return DataPreprocessor()


class TestDataPreprocessorIntegration:
    """
    Интеграционные тесты полного пайплайна препроцессинга
    """

    def test_full_preprocessing_pipeline(self, preprocessor: DataPreprocessor,
                                        raw_data_path: Path,
                                        tmp_path_factory) -> None:
        """
        Проверяет полный цикл препроцессинга: загрузка, преобразование,
        сохранение
        Args:
            preprocessor: экземпляр препроцессора
            raw_data_path: путь к сырым данным
            tmp_path_factory: фабрика временных директорий
        """
        output_dir = tmp_path_factory.mktemp('processed')
        output_path = output_dir / 'processed.csv'
        scaler_path = output_dir / 'scaler.pkl'
        df_processed = preprocessor.process(
            input_path=raw_data_path, output_path=output_path,
            save_scaler=True, scaler_path=scaler_path)
        assert output_path.exists()
        assert scaler_path.exists()
        expected_cols = [
            'it_density', 'sec_effectiveness', 'downtime_cap',
            'industry_finance', 'industry_healthcare', 'industry_it_telecom',
            'industry_manufacturing', 'industry_other', 'industry_retail',
            'T1', 'T2', 'T3', 'T4'
        ]
        for col in expected_cols:
            assert col in df_processed.columns, f"Колонка {col} отсутствует"
        df_loaded = load_dataframe(output_path)
        pd.testing.assert_frame_equal(df_processed, df_loaded)
        scaler_params = load_pickle(scaler_path)
        assert isinstance(scaler_params, dict)
        assert 'scaler' in scaler_params
        assert isinstance(scaler_params['scaler'], StandardScaler)
        assert 'feature_columns' in scaler_params
        assert isinstance(scaler_params['feature_columns'], list)

    def test_engineered_features(self, preprocessor: DataPreprocessor,
                                 raw_data_path: Path) -> None:
        """
        Проверяет корректность вычисления инженерных признаков
        Args:
            preprocessor: экземпляр препроцессора
            raw_data_path: путь к сырым данным
        """
        df_raw = load_dataframe(raw_data_path)
        df_eng = preprocessor.feature_engineer.fit_transform(df_raw)
        assert 'it_density' in df_eng.columns
        assert 'sec_effectiveness' in df_eng.columns
        assert 'downtime_cap' in df_eng.columns
        servers = df_raw['Количество серверов']
        employees = df_raw['Количество сотрудников']
        expected_it_density = servers / (employees + 1e-6)
        np.testing.assert_allclose(
            df_eng['it_density'].values, expected_it_density.values,
            rtol=1e-5, err_msg='it_density вычислен неверно')
        security_spend = df_raw[
            'Расходы на кибербезопасность за последний год (млн руб.)']
        revenue = df_raw['Годовая выручка (млн руб.)']
        expected_sec_eff = security_spend / (revenue + 1e-6)
        np.testing.assert_allclose(
            df_eng['sec_effectiveness'].values, expected_sec_eff.values,
            rtol=1e-5, err_msg='sec_effectiveness вычислен неверно')
        downtime_loss = df_raw['Потери от часа простоя (млн руб.)']
        mtd = df_raw[
            'Максимально допустимое время простоя критических сервисов (в часах)']
        expected_downtime_cap = downtime_loss * mtd
        expected_downtime_cap = expected_downtime_cap.fillna(0)
        np.testing.assert_allclose(
            df_eng['downtime_cap'].values, expected_downtime_cap.values,
            rtol=1e-5, err_msg='downtime_cap вычислен неверно')

    def test_categorical_encoding(self, preprocessor: DataPreprocessor,
                                  raw_data_path: Path) -> None:
        """
        Проверяет One-Hot кодирование отрасли
        Args:
            preprocessor: экземпляр препроцессора
            raw_data_path: путь к сырым данным
        """
        df_raw = load_dataframe(raw_data_path)
        df_encoded = preprocessor.encoder.fit_transform(df_raw)
        assert 'Отрасль' not in df_encoded.columns
        industry_cols = [
            col for col in df_encoded.columns if col.startswith('industry_')]
        assert len(industry_cols) == len(preprocessor.encoder.categories)
        row_sums = df_encoded[industry_cols].sum(axis=1)
        assert (row_sums == 1).all()

    def test_scaling(self, preprocessor: DataPreprocessor,
                     raw_data_path: Path) -> None:
        """
        Проверяет, что масштабирование приводит к среднему ~0 и std ~1
        Args:
            preprocessor: экземпляр препроцессора
            raw_data_path: путь к сырым данным
        """
        df_raw = load_dataframe(raw_data_path)
        df_scaled = preprocessor.scaler.fit_transform(df_raw)
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
        numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        exclude_set = set(binary_cols + target_cols)
        scaled_cols = [col for col in numeric_cols if col not in exclude_set]
        for col in scaled_cols:
            if col in df_scaled.columns:
                mean = df_scaled[col].mean()
                std = df_scaled[col].std()
                assert abs(mean) < MEAN_TOLERANCE, \
                    f"Среднее {col} = {mean:.4f}, ожидается 0"
                assert abs(std - 1.0) < STD_TOLERANCE, \
                    f"Стандартное отклонение {col} = {std:.4f}, ожидается 1"

    def test_target_creation(self, preprocessor: DataPreprocessor,
                             raw_data_path: Path) -> None:
        """
        Проверяет корректность создания целевых переменных T1-T4
        Args:
            preprocessor: экземпляр препроцессора
            raw_data_path: путь к сырым данным
        """
        df_raw = load_dataframe(raw_data_path)
        df_with_targets = preprocessor.target_builder.build(df_raw)
        assert 'T1' in df_with_targets.columns
        assert 'T2' in df_with_targets.columns
        assert 'T3' in df_with_targets.columns
        assert 'T4' in df_with_targets.columns
        y_true = df_raw[
            'Фактический совокупный ущерб за последние 12 месяцев (млн руб.)']
        expected_t1 = (y_true > 0).astype(float)
        pd.testing.assert_series_equal(
            df_with_targets['T1'], expected_t1, check_names=False)
        expected_t2 = np.log1p(y_true)
        expected_t2[y_true == 0] = np.nan
        np.testing.assert_allclose(
            df_with_targets['T2'].values, expected_t2.values,
            rtol=1e-6, equal_nan=True)
        expected_t3 = np.log1p(y_true)
        np.testing.assert_allclose(
            df_with_targets['T3'].values, expected_t3.values, rtol=1e-6)
        expected_t4 = y_true.quantile(0.90)
        assert (df_with_targets['T4'] == expected_t4).all(), \
            f"T4 должна быть константой {expected_t4:.4f}"

    def test_save_and_load_scaler(self, preprocessor: DataPreprocessor,
                                  raw_data_path: Path,
                                  tmp_path_factory) -> None:
        """
        Проверяет сохранение и загрузку scaler
        Args:
            preprocessor: экземпляр препроцессора
            raw_data_path: путь к сырым данным
            tmp_path_factory: фабрика временных директорий
        """
        df_raw = load_dataframe(raw_data_path)
        preprocessor.scaler.fit(df_raw)
        tmp_dir = tmp_path_factory.mktemp('scaler_test')
        scaler_path = tmp_dir / 'scaler.pkl'
        scaler_params = preprocessor.scaler.get_scaler_params()
        save_pickle(scaler_params, scaler_path)
        new_preprocessor = DataPreprocessor()
        new_preprocessor.load_scaler(scaler_path)
        df_raw2 = load_dataframe(raw_data_path)
        df_transformed1 = preprocessor.scaler.transform(df_raw2)
        df_transformed2 = new_preprocessor.scaler.transform(df_raw2)
        pd.testing.assert_frame_equal(
            df_transformed1, df_transformed2, check_dtype=False)

    def test_no_nan_after_preprocessing(self, preprocessor: DataPreprocessor,
                                        raw_data_path: Path) -> None:
        """
        Проверяет, что после препроцессинга нет NaN и inf
        (кроме T2, где NaN допустим)
        Args:
            preprocessor: экземпляр препроцессора
            raw_data_path: путь к сырым данным
        """
        df_raw = load_dataframe(raw_data_path)
        df_processed = preprocessor.feature_engineer.fit_transform(df_raw)
        df_processed = preprocessor.encoder.fit_transform(df_processed)
        df_processed = preprocessor.scaler.fit_transform(df_processed)
        df_processed = preprocessor.target_builder.build(df_processed)
        cols_to_check = [col for col in df_processed.columns if col != 'T2']
        df_subset = df_processed[cols_to_check]
        has_nan = df_subset.isnull().any().any()
        assert not has_nan, "NaN обнаружены в обработанных данных (кроме T2)"
        numeric_values = df_subset.select_dtypes(include=[np.number]).values
        has_inf = np.isinf(numeric_values).any()
        assert not has_inf, "inf обнаружены в обработанных данных"
