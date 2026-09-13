"""
Сервис запуска pytest
"""

from cli.utils import ScriptRunner


class TestRunner:
    """
    Запускает тесты через pytest и возвращает код возврата
    """

    def run_all(self) -> int:
        """
        Запускает все тесты
        Returns:
            Код возврата pytest
        """
        return ScriptRunner.run_pytest('tests', ['-v'])

    def run_module(self, path: str) -> int:
        """
        Запускает один тестовый модуль
        Args:
            path: путь к модулю
        Returns:
            Код возврата pytest
        """
        return ScriptRunner.run_pytest(path, ['-v'])
