"""
Утилиты для CLI: цветной вывод, запуск подпроцессов, форматирование
"""

import subprocess
import sys
from typing import List, Optional, Type


class Console:
    """
    Хелпер для цветного вывода в консоль
    """
    # ANSI-коды
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'

    @classmethod
    def title(cls: Type['Console'], text: str) -> None:
        """
        Печатает заголовок
        Args:
            text: текст заголовка
        """
        print(f"\n{cls.BOLD}{cls.CYAN}{text}{cls.RESET}\n")

    @classmethod
    def info(cls: Type['Console'], text: str) -> None:
        """
        Печатает информационное сообщение
        Args:
            text: текст информационного сообщения
        """
        print(f"{cls.BLUE}ℹ{cls.RESET} {text}")

    @classmethod
    def success(cls: Type['Console'], text: str) -> None:
        """
        Печатает сообщение об успехе
        Args:
            text: текст сообщения об успехе
        """
        print(f"{cls.GREEN}✓{cls.RESET} {text}")

    @classmethod
    def error(cls: Type['Console'], text: str) -> None:
        """
        Печатает сообщение об ошибке
        Args:
            text: текст сообщения об ошибке
        """
        print(f"{cls.RED}✗{cls.RESET} {text}")

    @classmethod
    def warning(cls: Type['Console'], text: str) -> None:
        """
        Печатает предупреждение
        Args:
            text: текст предупреждения
        """
        print(f"{cls.YELLOW}⚠{cls.RESET} {text}")

    @classmethod
    def muted(cls: Type['Console'], text: str) -> None:
        """
        Печатает приглушенный текст
        Args:
            text: приглушенный текст
        """
        print(f"{cls.DIM}{text}{cls.RESET}")


class ScriptRunner:
    """
    Запускает python-скрипты проекта как подпроцессы
    """

    @staticmethod
    def run(script: str, args: Optional[List[str]] = None) -> int:
        """
        Запускает скрипт через текущий интерпретатор
        Args:
            script: путь к скрипту от корня проекта
            args: дополнительные аргументы командной строки
        Returns:
            Код возврата подпроцесса (0 - успех)
        """
        cmd = [sys.executable, script] + (args or [])
        Console.info(f"Запуск: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=False)
            return result.returncode
        except KeyboardInterrupt:
            Console.warning("Процесс прерван пользователем")
            return 130
        except FileNotFoundError:
            Console.error(f"Скрипт не найден: {script}")
            return 1

    @staticmethod
    def run_pytest(path: str = 'tests', extra: Optional[List[str]] = None) -> int:
        """
        Запускает pytest
        Args:
            path: путь к тестам
            extra: дополнительные аргументы (например, ['-v', '-k', 'infer'])
        Returns:
            Код возврата pytest
        """
        cmd = [sys.executable, '-m', 'pytest', path] + (extra or [])
        Console.info(f"Запуск: {' '.join(cmd)}")
        return subprocess.run(cmd, check=False).returncode
