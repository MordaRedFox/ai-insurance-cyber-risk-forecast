"""
Утилиты для работы с файлами
"""

import pickle
import pandas as pd
from pathlib import Path
from typing import Union, Any


def save_dataframe(df: pd.DataFrame, filepath: Union[str, Path],
                   index: bool = False) -> None:
    """
    Сохраняет pandas DataFrame в CSV или Parquet
    Args:
        df: датафрейм для сохранения
        filepath: путь с расширением .csv или .parquet
        index: сохранять ли индекс
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.suffix == '.csv':
        df.to_csv(filepath, index=index)
    elif filepath.suffix == '.parquet':
        df.to_parquet(filepath, index=index)
    else:
        raise ValueError(
            "Неподдерживаемое расширение. Используйте .csv или .parquet")

def load_dataframe(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Загружает DataFrame из CSV или Parquet
    Args:
        filepath: путь к файлу с расширением .csv или .parquet 
    Returns:
        Загруженный DataFrame
    """
    filepath = Path(filepath)
    if filepath.suffix == '.csv':
        return pd.read_csv(filepath)
    elif filepath.suffix == '.parquet':
        return pd.read_parquet(filepath)
    else:
        raise ValueError(
            "Неподдерживаемое расширение. Используйте .csv или .parquet")

def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """
    Сохраняет Python-объект в файл с помощью pickle
    Args:
        obj: любой сериализуемый объект
        filepath: путь для сохранения (расширение обычно .pkl)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Загружает Python-объект из pickle-файла
    Args:
        filepath: путь к файлу .pkl
    Returns:
        Восстановленный объект
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)
