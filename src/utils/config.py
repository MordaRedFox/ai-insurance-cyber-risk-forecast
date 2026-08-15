"""
Утилиты для работы с конфигами
"""

import yaml
from pathlib import Path
from typing import Union


def load_config(config_path: Union[str, Path]) -> dict:
    """
    Загружает YAML-конфигурационный файл
    Args:
        config_path: путь к файлу .yaml или .yml
    Returns:
        Словарь с параметрами конфигурации
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config: dict, path: Union[str, Path]) -> None:
    """
    Сохраняет словарь в YAML-файл
    Args:
        config: словарь конфигурации
        path: путь для сохранения
    """
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
