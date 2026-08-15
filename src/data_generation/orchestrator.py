"""
Оркестратор, объединяющий все генераторы и создающий итоговый DataFrame
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Union
from src.utils import set_seed, load_config, save_dataframe
from src.data_generation.industry import IndustryGenerator
from src.data_generation.continuous import ContinuousFeatureGenerator
from src.data_generation.share import ShareFeatureGenerator
from src.data_generation.binary import BinaryFeatureGenerator
from src.data_generation.pentest import PentestGenerator
from src.data_generation.security import SecuritySpendGenerator
from src.data_generation.target import TargetGenerator

logger = logging.getLogger(__name__)


class DataGenerator:
    """
    Главный класс, который управляет всеми генераторами и собирает датасет
    """

    def __init__(self, config_path: Union[str, Path]):
        """
        Инициализация генератора данных
        Args:
            config_path: путь к YAML-файлу с конфигурацией генерации
        """
        self.config = load_config(config_path)
        self._validate_config()
        self.seed = self.config['seed']
        set_seed(self.seed)
        self.rng = np.random.RandomState(self.seed)
        self.num_samples = self.config['num_samples']

    def _validate_config(self) -> None:
        """
        Проверяет наличие всех обязательных разделов в конфигурационном словаре
        """
        required_keys = [
            'seed', 'num_samples', 'industry_probs', 'log_normal_params',
            'beta_params', 'binary_probs', 'pentest_probs',
            'correlation_matrix', 'security_spend', 'target_coefficients',
            'noise_std', 'zero_loss_prob'
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(
                    f"Отсутствует необходимый ключ конфигурации: {key}")

    def generate_dataset(self) -> pd.DataFrame:
        """
        Основной метод, создающий полный датасет
        Returns:
            pd.DataFrame: полный датасет с признаками и целевой переменной
        """
        logger.info(
            f"Генерирование {self.num_samples} синтетических компаний...")
        # 1. Отрасль
        industry_gen = IndustryGenerator(self.config, self.rng)
        industries = industry_gen.generate()
        # 2. Непрерывные признаки
        cont_gen = ContinuousFeatureGenerator(self.config, self.rng)
        df_cont = cont_gen.generate()
        # 3. Расходы на безопасность
        security_gen = SecuritySpendGenerator(self.config, self.rng)
        security_spend = security_gen.generate(
            df_cont['revenue'].values, industries)
        df_cont['security_spend'] = security_spend
        # 4. Доли
        share_gen = ShareFeatureGenerator(self.config, self.rng)
        df_shares = share_gen.generate(industries)
        # 5. Бинарные
        binary_gen = BinaryFeatureGenerator(self.config, self.rng)
        df_binary = binary_gen.generate(industries)
        # 6. Пентест
        pentest_gen = PentestGenerator(self.config, self.rng)
        pentest = pentest_gen.generate(industries)
        # 7. Признаки простоя
        df_downtime = self._generate_downtime_features()
        # 8. Объединение
        df_features = pd.concat([df_cont, df_shares, df_binary,
            pd.DataFrame({'pentest': pentest}), df_downtime], axis=1)
        df_features['industry'] = industries
        # 9. Целевая переменная
        target_gen = TargetGenerator(self.config, self.rng)
        y = target_gen.generate(df_features)
        df_features['actual_loss'] = y
        # 10. Переименование колонок согласно документации
        df_features = self._rename_columns(df_features)
        # 11. Приведение типов бинарных колонок
        self._cast_binary_columns(df_features)
        return df_features

    def _generate_downtime_features(self) -> pd.DataFrame:
        """
        Генерирует признаки простоя (лог-нормальные, независимые)
        Returns:
            pd.DataFrame: с двумя колонками, каждая размером (num_samples,)
        """
        log_params = self.config['log_normal_params']
        names = ['downtime_hour_loss', 'mtd_hours']
        means = [log_params['downtime_hour_loss']['mean'],
                 log_params['mtd_hours']['mean']]
        stds = [log_params['downtime_hour_loss']['std'],
                log_params['mtd_hours']['std']]
        log_vals = self.rng.normal(
            loc=means, scale=stds, size=(self.num_samples, 2))
        vals = np.exp(log_vals)
        return pd.DataFrame(vals, columns=names)

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Переименовывает колонки в человекочитаемые названия
        Args:
            df: исходный DataFrame с английскими названиями колонок
        Returns:
            pd.DataFrame: с переименованными колонками (на русском языке)
        """
        mapping = {
            'revenue': 'Годовая выручка (млн руб.)',
            'intangible_assets': 'Стоимость нематериальных активов (млн руб.)',
            'employees': 'Количество сотрудников',
            'online_share': 'Доля онлайн-продаж',
            'security_spend': 'Расходы на кибербезопасность за последний год (млн руб.)',
            'servers': 'Количество серверов',
            'cloud_share': 'Доля облачных сервисов в ИТ-инфраструктуре',
            'critical_apps': 'Количество критических бизнес-приложений',
            'iot_ot': 'Количество подключенных IoT/OT-устройств',
            'has_policy': 'Наличие формальной политики безопасности',
            'pentest': 'Периодичность внешнего пентеста',
            'has_soc_siem': 'Наличие SOC / SIEM-системы и 24/7 мониторинга',
            'training_percent': 'Процент сотрудников, прошедших обучение кибергигиене за год',
            'has_irp': 'Наличие плана реагирования на инциденты (IRP)',
            'industry': 'Отрасль',
            'personal_data': 'Объем обрабатываемых персональных данных (количество записей, млн)',
            'incidents_3y': 'Количество киберинцидентов за последние 3 года',
            'past_loss': 'Суммарный ущерб от инцидентов за последние 3 года (млн руб.)',
            'has_insurance': 'Наличие страхового покрытия киберрисков в прошлом',
            'past_payout': 'Размер страховых выплат в прошлом (млн руб.)',
            'has_network_segmentation': 'Наличие сегментации сети',
            'has_eol_software': 'Наличие устаревшего ПО/EOL',
            'downtime_hour_loss': 'Потери от часа простоя (млн руб.)',
            'mtd_hours': 'Максимально допустимое время простоя критических сервисов (в часах)',
            'actual_loss': 'Фактический совокупный ущерб за последние 12 месяцев (млн руб.)'
        }
        return df.rename(columns=mapping)

    def _cast_binary_columns(self, df: pd.DataFrame) -> None:
        """
        Приводит бинарные колонки к типу int (изменяет DataFrame на месте)
        Args:
            df: DataFrame, содержащий бинарные колонки
        """
        binary_cols = [
            'Наличие формальной политики безопасности',
            'Наличие SOC / SIEM-системы и 24/7 мониторинга',
            'Наличие плана реагирования на инциденты (IRP)',
            'Наличие страхового покрытия киберрисков в прошлом',
            'Наличие сегментации сети',
            'Наличие устаревшего ПО/EOL'
        ]
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)

    def save_dataset(self, df: pd.DataFrame,
                     output_path: Union[str, Path]) -> None:
        """
        Сохраняет сгенерированный датасет в файл (CSV или Parquet)
        Args:
            df: DataFrame для сохранения
            output_path: путь к файлу с расширением .csv или .parquet
        """
        save_dataframe(df, output_path)
        logger.info(f"Набор данных сохранен в {output_path}")
