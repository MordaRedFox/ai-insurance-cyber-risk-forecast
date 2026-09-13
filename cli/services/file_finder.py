"""
Сервис поиска файлов по расширению
"""

from pathlib import Path
from typing import List


class FileFinder:
    """
    Ищет файлы с заданным расширением в указанной папке
    """

    def find(self, folder: Path, extension: str) -> List[Path]:
        """
        Возвращает отсортированный список файлов с заданным расширением
        Args:
            folder: папка для поиска
            extension: расширение файлов (например, '*.png')
        Returns:
            Список путей к найденным файлам, отсортированный по имени
        """
        if not folder.exists():
            return []
        return sorted(folder.glob(extension))
