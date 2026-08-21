"""
Утилиты для настройки логирования
"""

import logging
import sys
import io
from pathlib import Path
from datetime import datetime
from typing import Union


def setup_logging(log_dir: Union[str, Path] = 'logs',
                  name: str = 'project') -> logging.Logger:
    """
    Настройка логирования: создает папки для логов и файла с временной меткой
    Использует кодировку UTF-8 для консоли и файлов
    Args:
        log_dir: директория для сохранения лог-файлов
        name: префикс имени лог-файла
    Returns:
        Настроенный объект logger
    """
    if hasattr(sys.stdout, 'buffer') and not isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding='utf-8', errors='replace')
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
