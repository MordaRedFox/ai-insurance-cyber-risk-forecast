"""
Команды запуска тестов
"""

from typing import Optional
from cli.commands.base import BaseCommand
from cli.services import TestModule, TestRegistry, TestRunner
from cli.utils import Console


SEPARATOR_WIDTH = 52


class BaseTestCommand(BaseCommand):
    """
    Общий предок для команд запуска тестов
    Содержит логику, специфичную только для тестовых команд
    """

    def __init__(self):
        """
        Инициализирует сервис запуска тестов
        """
        self.runner = TestRunner()

    @staticmethod
    def print_test_result(code: int) -> None:
        """
        Печатает результат запуска тестов в консоль
        Args:
            code: код возврата pytest
        """
        if code == 0:
            Console.success("Все тесты прошли.")
        else:
            Console.error(f"Тесты завершились с кодом {code}.")


class RunAllTestsCommand(BaseTestCommand):
    """
    Запускает все тесты
    """
    name = 'Все тесты'
    description = 'Запустить полный набор pytest'

    def execute(self) -> None:
        """
        Запускает все тесты сразу
        """
        Console.title("Запуск всех тестов")
        code = self.runner.run_all()
        self.print_test_result(code)


class RunTestModuleCommand(BaseTestCommand):
    """
    Запускает один тестовый модуль по выбору
    """
    name = 'Один тестовый модуль'
    description = 'Выбрать конкретный модуль тестов'
    QUIT_KEYS = {'q', 'quit', 'exit', '0'}

    def __init__(self):
        """
        Инициализирует реестр модулей
        """
        super().__init__()
        self.registry = TestRegistry()

    def execute(self) -> None:
        """
        Показывает меню модулей и запускает выбранный
        """
        while True:
            self._print_menu()
            choice = self._read_choice()
            if choice is None:
                return
            if choice in self.QUIT_KEYS:
                Console.info("Возврат в главное меню.")
                return
            module = self.registry.find(choice)
            if module is None:
                Console.error("Некорректный выбор. Попробуйте снова.")
                continue
            self._run_module(module)
            return

    def _print_menu(self) -> None:
        """
        Отображает список тестовых модулей в стиле главного меню
        """
        Console.title("Выбор тестового модуля")
        print("─" * SEPARATOR_WIDTH)
        for module in self.registry.all():
            print(f"  {Console.GREEN}{module.key}{Console.RESET}. "
                  f"{Console.BOLD}{module.label}{Console.RESET}")
        print("─" * SEPARATOR_WIDTH)
        Console.muted("  q. Возврат в главное меню")

    @staticmethod
    def _read_choice() -> Optional[str]:
        """
        Читает выбор пользователя
        Returns:
            Строка выбора или None при прерывании
        """
        try:
            prompt = f"\n{Console.YELLOW}➜{Console.RESET}  Выбор: "
            return input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

    def _run_module(self, module: TestModule) -> None:
        """
        Запускает выбранный модуль
        Args:
            module: тестовый модуль
        """
        Console.info(f"Запуск: {module.label}")
        code = self.runner.run_module(module.path)
        self.print_test_result(code)
