"""
Скрипт для запуска препроцессинга данных
"""

import sys
sys.path.append('.')
from src import DataPreprocessor, setup_logging

logger = setup_logging(log_dir='logs', name='preprocessing')


if __name__ == '__main__':
    preprocessor = DataPreprocessor()
    input_path = 'data/raw/synthetic_companies.csv'
    output_path = 'data/processed/processed_companies.csv'
    scaler_path = 'models/scaler.pkl'
    df = preprocessor.process(input_path=input_path, output_path=output_path,
                              save_scaler=True, scaler_path=scaler_path)
    logger.info(f"Обработано {len(df)} записей")
    logger.info(f"Колонки: {df.columns.tolist()}")
    logger.info(f"Целевые T1: {df['T1'].sum()} инцидентов ({
        df['T1'].mean()*100:.1f}%)")
    logger.info(f"T2 (средний лог-ущерб): {df['T2'].mean():.2f}")
    logger.info(f"T3 (средний лог-ущерб): {df['T3'].mean():.2f}")
