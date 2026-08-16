"""
Генератор целевой переменной Y0 (фактический ущерб)
"""

import numpy as np
import pandas as pd
from src.data_generation.base import BaseGenerator


class TargetGenerator(BaseGenerator):
    """
    Генерирует Y0 на основе линейной комбинации признаков, шума и обнуления
    """

    def generate(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Генерирует Y0 на основе линейной комбинации признаков, шума и обнуления
        Args:
            df_features: полный DataFrame со всеми признаками
        Returns:
            Массив значений ущерба (млн руб.) размером (num_samples,)
        """
        coef = self.config['target_coefficients']
        noise_std = self.config['noise_std']
        zero_prob = self.config['zero_loss_prob']
        df = df_features.copy()
        # Логарифмирование нужных колонок
        log_cols = [
            'revenue', 'intangible_assets', 'employees', 'servers',
            'critical_apps', 'iot_ot', 'personal_data', 'incidents_3y',
            'past_loss', 'past_payout', 'downtime_hour_loss', 'mtd_hours'
        ]
        for col in log_cols:
            df[col + '_log'] = np.log1p(df[col])
        # One-hot encoding отраслей
        industry_dummies = pd.get_dummies(df['industry'], prefix='industry')
        # Формирование матрицы признаков X
        X = pd.DataFrame()
        X['intercept'] = 1.0
        # Основные признаки
        X['beta_revenue'] = df['revenue_log']
        X['beta_intangible'] = df['intangible_assets_log']
        X['beta_employees'] = df['employees_log']
        X['beta_online_share'] = df['online_share']
        X['beta_security_spend'] = df['security_spend']
        X['beta_servers'] = df['servers_log']
        X['beta_cloud_share'] = df['cloud_share']
        X['beta_critical_apps'] = df['critical_apps_log']
        X['beta_iot_ot'] = df['iot_ot_log']
        X['beta_has_policy'] = df['has_policy']
        X['beta_pentest'] = df['pentest']
        X['beta_has_soc'] = df['has_soc_siem']
        X['beta_training'] = df['training_percent']
        X['beta_has_irp'] = df['has_irp']
        # Отраслевые признаки
        for ind in self.industry_names:
            if ind != 'other':
                X[f'industry_{ind}'] = industry_dummies[f'industry_{ind}']
        X['beta_personal_data'] = df['personal_data_log']
        X['beta_incidents_3y'] = df['incidents_3y_log']
        X['beta_past_loss'] = df['past_loss_log']
        X['beta_past_payout'] = df['past_payout_log']
        X['beta_has_insurance'] = df['has_insurance']
        X['beta_network_seg'] = df['has_network_segmentation']
        X['beta_has_eol'] = df['has_eol_software']
        X['beta_downtime_hour_loss'] = df['downtime_hour_loss_log']
        X['beta_mtd'] = df['mtd_hours_log']
        # Взаимодействия
        X['beta_interaction_eol_servers'] = df[
            'has_eol_software'] * df['servers_log']
        X['beta_interaction_online_revenue'] = df[
            'online_share'] * df['revenue_log']
        X['beta_interaction_downtime_mtd'] = df[
            'downtime_hour_loss_log'] * df['mtd_hours_log']
        X['beta_interaction_seg_servers'] = df[
            'has_network_segmentation'] * df['servers_log']
        X = X.astype(np.float64)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # Список коэффициентов в правильном порядке
        coef_names = [
            'intercept', 'beta_revenue', 'beta_intangible', 'beta_employees',
            'beta_online_share', 'beta_security_spend', 'beta_servers',
            'beta_cloud_share', 'beta_critical_apps', 'beta_iot_ot',
            'beta_has_policy', 'beta_pentest', 'beta_has_soc',
            'beta_training', 'beta_has_irp', 'industry_finance',
            'industry_retail', 'industry_manufacturing',
            'industry_healthcare', 'industry_it_telecom',
            'beta_personal_data', 'beta_incidents_3y', 'beta_past_loss',
            'beta_past_payout', 'beta_has_insurance', 'beta_network_seg',
            'beta_has_eol', 'beta_downtime_hour_loss', 'beta_mtd',
            'beta_interaction_eol_servers', 'beta_interaction_online_revenue',
            'beta_interaction_downtime_mtd', 'beta_interaction_seg_servers'
        ]
        betas = np.array([coef[name] for name in coef_names], dtype=np.float64)
        X_ordered = X[coef_names].values.astype(np.float64)
        linear_pred = np.dot(X_ordered, betas)
        noise = self.rng.normal(
            0, noise_std, size=self.num_samples).astype(np.float64)
        log_y = linear_pred + noise
        y = np.exp(log_y)
        # Обнуление части компаний
        zero_mask = self.rng.binomial(
            1, zero_prob, size=self.num_samples).astype(bool)
        y[zero_mask] = 0.0
        return y
