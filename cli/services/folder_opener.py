"""
Сервис открытия папок в системном файловом менеджере
"""

import os
import platform
import subprocess
from pathlib import Path


class FolderOpener:
    """
    Открывает папку в файловом менеджере текущей ОС
    Поддерживает Windows (os.startfile), macOS (open), Linux (xdg-open)
    """

    def open(self, folder: Path) -> None:
        """
        Открывает папку в системном файловом менеджере
        Args:
            folder: путь к папке
        """
        folder = folder.resolve()
        system = platform.system()
        try:
            if system == 'Windows':
                os.startfile(str(folder))
            elif system == 'Darwin':
                subprocess.run(['open', str(folder)], check=False)
            else:
                subprocess.run(['xdg-open', str(folder)], check=False)
        except OSError as exc:
            raise RuntimeError(
                f"Не удалось открыть папку {folder}: {exc}") from exc
