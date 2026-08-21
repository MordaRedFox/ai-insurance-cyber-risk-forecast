"""
Класс Trainer для управления обучением модели
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any
from src.train.losses import pinball_loss

logger = logging.getLogger(__name__)


class Trainer:
    """
    Управляет процессом обучения модели
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, config: Dict[str, Any],
                 device: torch.device):
        """
        Инициализирует тренера
        Args:
            model: экземпляр модели
            train_loader: DataLoader для обучающей выборки
            val_loader: DataLoader для валидационной выборки
            config: словарь с конфигурацией обучения
            device: устройство
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.get('scheduler_factor', 0.5),
            patience=config.get('scheduler_patience', 5)
        )
        self.criterion_t1 = nn.BCEWithLogitsLoss()
        self.criterion_t2 = nn.MSELoss()
        self.criterion_t3 = nn.MSELoss()
        self.best_val_loss = float('inf')
        self.patience = config.get('early_stopping_patience', 10)
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self) -> Dict[str, float]:
        """
        Обучает модель на одной эпохе
        Returns:
            Словарь с усредненными потерями: total_loss, t1, t2, t3, t4
        """
        self.model.train()
        total_loss = 0.0
        total_t1 = 0.0
        total_t2 = 0.0
        total_t3 = 0.0
        total_t4 = 0.0
        num_batches = 0
        for batch in self.train_loader:
            X, y1, y2, y3, y4 = batch
            X = X.to(self.device)
            y1 = y1.to(self.device).float().view(-1, 1)
            y2 = y2.to(self.device).float().view(-1, 1)
            y3 = y3.to(self.device).float().view(-1, 1)
            y4 = y4.to(self.device).float().view(-1, 1)
            self.optimizer.zero_grad()
            outputs = self.model(X)
            t1_logits = outputs['t1']
            t2_pred = outputs['t2']
            t3_pred = outputs['t3']
            t4_pred = outputs['t4']
            loss_t1 = self.criterion_t1(t1_logits, y1)
            mask_t2 = (y1 > 0.5).squeeze()
            if mask_t2.sum() > 0:
                loss_t2 = self.criterion_t2(t2_pred[mask_t2], y2[mask_t2])
            else:
                loss_t2 = torch.tensor(0.0, device=self.device)
            loss_t3 = self.criterion_t3(t3_pred, y3)
            loss_t4 = pinball_loss(t4_pred, y4, tau=0.9)
            w1 = self.config.get('loss_weights', {}).get('t1', 1.0)
            w2 = self.config.get('loss_weights', {}).get('t2', 1.0)
            w3 = self.config.get('loss_weights', {}).get('t3', 1.0)
            w4 = self.config.get('loss_weights', {}).get('t4', 1.0)
            loss = w1 * loss_t1 + w2 * loss_t2 + w3 * loss_t3 + w4 * loss_t4
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            total_t1 += loss_t1.item()
            total_t2 += loss_t2.item()
            total_t3 += loss_t3.item()
            total_t4 += loss_t4.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        avg_t1 = total_t1 / num_batches
        avg_t2 = total_t2 / num_batches
        avg_t3 = total_t3 / num_batches
        avg_t4 = total_t4 / num_batches
        logger.info(
            f"Train Loss: {avg_loss:.4f} (t1: {avg_t1:.4f}, t2: {avg_t2:.4f}, t3: {avg_t3:.4f}, t4: {avg_t4:.4f})"
        )
        return {
            'total_loss': avg_loss,
            't1': avg_t1,
            't2': avg_t2,
            't3': avg_t3,
            't4': avg_t4
        }

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Оценивает модель на валидационном или тестовом DataLoader
        Args:
            data_loader: DataLoader, содержащий батчи с данными
        Returns:
            Словарь с ключами: total_loss, t1, t2, t3, t4 (усредненные по батчам)
        """
        self.model.eval()
        total_loss = 0.0
        total_t1 = 0.0
        total_t2 = 0.0
        total_t3 = 0.0
        total_t4 = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in data_loader:
                X, y1, y2, y3, y4 = batch
                X = X.to(self.device)
                y1 = y1.to(self.device).float().view(-1, 1)
                y2 = y2.to(self.device).float().view(-1, 1)
                y3 = y3.to(self.device).float().view(-1, 1)
                y4 = y4.to(self.device).float().view(-1, 1)
                outputs = self.model(X)
                t1_logits = outputs['t1']
                t2_pred = outputs['t2']
                t3_pred = outputs['t3']
                t4_pred = outputs['t4']
                loss_t1 = self.criterion_t1(t1_logits, y1)
                mask_t2 = (y1 > 0.5).squeeze()
                if mask_t2.sum() > 0:
                    loss_t2 = self.criterion_t2(t2_pred[mask_t2], y2[mask_t2])
                else:
                    loss_t2 = torch.tensor(0.0, device=self.device)
                loss_t3 = self.criterion_t3(t3_pred, y3)
                loss_t4 = pinball_loss(t4_pred, y4, tau=0.9)
                w1 = self.config.get('loss_weights', {}).get('t1', 1.0)
                w2 = self.config.get('loss_weights', {}).get('t2', 1.0)
                w3 = self.config.get('loss_weights', {}).get('t3', 1.0)
                w4 = self.config.get('loss_weights', {}).get('t4', 1.0)
                loss = w1 * loss_t1 + w2 * loss_t2 + w3 * loss_t3 + w4 * loss_t4
                total_loss += loss.item()
                total_t1 += loss_t1.item()
                total_t2 += loss_t2.item()
                total_t3 += loss_t3.item()
                total_t4 += loss_t4.item()
                num_batches += 1
        avg_loss = total_loss / num_batches
        avg_t1 = total_t1 / num_batches
        avg_t2 = total_t2 / num_batches
        avg_t3 = total_t3 / num_batches
        avg_t4 = total_t4 / num_batches
        logger.info(
            f"Val Loss: {avg_loss:.4f} (t1: {avg_t1:.4f}, t2: {avg_t2:.4f}, t3: {avg_t3:.4f}, t4: {avg_t4:.4f})"
        )
        return {
            'total_loss': avg_loss,
            't1': avg_t1,
            't2': avg_t2,
            't3': avg_t3,
            't4': avg_t4
        }

    def train(self, num_epochs: int, save_dir: Path,
              model_name: str = 'best_model.pth') -> nn.Module:
        """
        Запускает полный цикл обучения
        Args:
            num_epochs: максимальное число эпох
            save_dir: директория для сохранения модели
            model_name: имя файла для лучшей модели
        Returns:
            Обученная модель
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            train_loss = train_metrics['total_loss']
            val_loss = val_metrics['total_loss']
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            logger.info(
                f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            self.scheduler.step(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), save_dir / model_name)
                logger.info(f"Модель сохранена с val loss {val_loss:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Ранняя остановка на эпохе {epoch}")
                    break
        logger.info("Обучение завершено")
        logger.info(f"Лучшая val loss: {self.best_val_loss:.4f}")
        self.model.load_state_dict(torch.load(save_dir / model_name))
        return self.model
