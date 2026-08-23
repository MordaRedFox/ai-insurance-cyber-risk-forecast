"""
Скрипт запуска обучения модели
"""

import sys
sys.path.append('.')
import logging
import torch
from pathlib import Path
from src.train import Trainer, prepare_dataloaders
from src.model import CyberRiskModel
from src.utils import load_config, set_seed, load_dataframe


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    model_config_path = 'configs/model_config.yaml'
    training_config_path = 'configs/training_config.yaml'
    model_config = load_config(model_config_path)
    training_config = load_config(training_config_path)
    set_seed(training_config.get('training', {}).get('random_state', 42))
    data_path = Path('data/processed/processed_companies.csv')
    df = load_dataframe(data_path)
    logger.info(f"Загружено {len(df)} записей")
    target_cols = ['T1', 'T2', 'T3']
    exclude_cols = target_cols + [
        'Фактический совокупный ущерб за последние 12 месяцев (млн руб.)']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    logger.info(f"Количество признаков: {len(feature_cols)}")
    logger.info(f"Пример признаков: {feature_cols[:5]}")
    batch_size = training_config.get('training', {}).get('batch_size', 256)
    val_split = training_config.get('training', {}).get('train_val_split', 0.8)
    random_state = training_config.get('training', {}).get('random_state', 42)
    train_loader, val_loader = prepare_dataloaders(
        df, feature_cols, target_cols, batch_size=batch_size,
        val_split=1 - val_split, random_state=random_state)
    input_dim = len(feature_cols)
    hidden_dims = model_config['model']['hidden_dims']
    dropout = model_config['model']['dropout']
    head_dropout = model_config['model'].get('head_dropout', 0.2)
    activation = model_config['model'].get('activation', 'relu')
    use_batch_norm = model_config['model'].get('use_batch_norm', True)
    model = CyberRiskModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        head_dropout=head_dropout,
        activation=activation,
        use_batch_norm=use_batch_norm
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используется устройство: {device}")
    train_config = {
        'learning_rate': float(training_config.get('training', {}).get(
            'learning_rate', 0.001)),
        'weight_decay': float(training_config.get('training', {}).get(
            'weight_decay', 1e-5)),
        'scheduler_patience': int(training_config.get('training', {}).get(
            'scheduler_patience', 5)),
        'scheduler_factor': float(training_config.get('training', {}).get(
            'scheduler_factor', 0.5)),
        'early_stopping_patience': int(training_config.get('training', {}).get(
            'early_stopping_patience', 10)),
        'loss_weights': model_config.get(
            'loss_weights', {})
    }
    trainer = Trainer(model, train_loader, val_loader, train_config, device)
    num_epochs = training_config.get(
        'training', {}).get('num_epochs', 100)
    save_dir = training_config.get(
        'logging', {}).get('save_dir', 'models/checkpoints/')
    model_name = 'best_model.pth'
    trainer.train(
        num_epochs=num_epochs, save_dir=save_dir, model_name=model_name)
    logger.info(
        f"Средние значения целей: T1={df['T1'].mean():.2f}, "
        f"T2={df['T2'].mean():.2f}, T3={df['T3'].mean():.2f}")
    logger.info("Обучение завершено успешно")
