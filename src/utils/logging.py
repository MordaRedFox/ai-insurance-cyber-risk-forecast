"""
Утилиты для настройки логирования
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Union


def setup_logging(log_dir: Union[str, Path] = 'logs',
                  name: str = 'project') -> logging.Logger:
    """
    Настройка логирования: создает папки для логов и файла с временной меткой
    Args:
        log_dir: директория для сохранения лог-файлов
        name: префикс имени лог-файла
    Returns:
        Настроенный объект logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
