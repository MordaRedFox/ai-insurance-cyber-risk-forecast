"""
Загрузчик данных
"""

import pandas as pd
from pathlib import Path
from typing import Union
from src.utils import load_dataframe


class DataLoader:
    """
    Загружает сырые данные из CSV/Parquet
    """

    def __init__(self, input_path: Union[str, Path]):
        """
        Инициализирует загрузчик
        Args:
            input_path: путь к файлу с данными
        """
        self.input_path = Path(input_path)

    def load(self) -> pd.DataFrame:
        """
        Загружает DataFrame из файла
        Returns:
            Загруженный DataFrame
        """
        return load_dataframe(self.input_path)
