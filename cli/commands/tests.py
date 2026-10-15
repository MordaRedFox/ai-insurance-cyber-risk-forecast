"""
Команда запуска тестов: все сразу или конкретный модуль
"""

from typing import Optional
from cli.commands.base import BaseCommand
from cli.services import TestModule, TestRegistry, TestRunner
from cli.utils import Console


SEPARATOR_WIDTH = 52


class TestsCommand(BaseCommand):
    """
    Запуск тестов в подменю: весь пайплайн или отдельный модуль
    """
    name = 'Запуск тестов'
    description = 'Запустить все тесты или конкретный модуль'
    QUIT_KEYS = {'q', 'quit', 'exit', '0'}
    ALL_TESTS_KEY = '1'
    FIRST_MODULE_INDEX = 2

    def __init__(self):
        """
        Инициализирует сервис запуска тестов и реестр модулей
        """
        self.runner = TestRunner()
        self.registry = TestRegistry()
        self._key_to_module: dict = {}

    def execute(self) -> None:
        """
        Показывает подменю тестов и запускает выбранный сценарий
        """
        while True:
            self._print_menu()
            choice = self._read_choice()
            if choice is None or choice in self.QUIT_KEYS:
                Console.info("Возврат в главное меню.")
                Console.mark_handled()
                return
            if choice == self.ALL_TESTS_KEY:
                self._run_all()
                return
            module = self._key_to_module.get(choice)
            if module is None:
                Console.error("Некорректный выбор. Попробуйте снова.")
                continue
            self._run_module(module)
            return

    def _print_menu(self) -> None:
        """
        Отображает подменю: «Все тесты», затем конкретные модули тестов
        """
        self._key_to_module = {}
        Console.title("Запуск тестов")
        print("─" * SEPARATOR_WIDTH)
        print(f"  {Console.GREEN}{self.ALL_TESTS_KEY}{Console.RESET}. "
              f"{Console.BOLD}Все тесты{Console.RESET}")
        for index, module in enumerate(
                self.registry.all(), start=self.FIRST_MODULE_INDEX):
            key = str(index)
            self._key_to_module[key] = module
            print(f"  {Console.GREEN}{key}{Console.RESET}. "
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

    def _run_all(self) -> None:
        """
        Запускает весь пайплайн тестов
        """
        Console.info("Запуск всех тестов...")
        code = self.runner.run_all()
        self._print_result(code)

    def _run_module(self, module: TestModule) -> None:
        """
        Запускает выбранный тестовый модуль
        Args:
            module: тестовый модуль из реестра
        """
        Console.info(f"Запуск: {module.label}")
        code = self.runner.run_module(module.path)
        self._print_result(code)

    @staticmethod
    def _print_result(code: int) -> None:
        """
        Печатает результат запуска тестов
        Args:
            code: код возврата pytest
        """
        if code == 0:
            Console.success("Все тесты прошли.")
        else:
            Console.error(f"Тесты завершились с кодом {code}.")
