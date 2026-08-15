"""
Скрипт для запуска генерации данных
"""

import sys
sys.path.append('.')
from src.data_generation import DataGenerator
from src.utils import setup_logging

if __name__ == "__main__":
    logger = setup_logging(log_dir='logs', name='generation')
    config_path = 'configs/generation_config.yaml'
    generator = DataGenerator(config_path)
    df = generator.generate_dataset()
    output_path = 'data/raw/synthetic_companies.csv'
    generator.save_dataset(df, output_path)
    # Вывод статистики
    target_col = 'Фактический совокупный ущерб за последние 12 месяцев (млн руб.)'
    logger.info(f"Сгенерировано {len(df)} рядов")
    logger.info(f"Целевые показатели Y0: min={df[target_col].min():.2f}, "
                f"max={df[target_col].max():.2f}, "
                f"mean={df[target_col].mean():.2f}, "
                f"median={df[target_col].median():.2f}")
    zero_count = (df[target_col] == 0).sum()
    zero_pct = zero_count / len(df) * 100
    logger.info(
        f"Компании с нулевыми убытками: {zero_count} ({zero_pct:.1f}%)")
    print("Готово.")
