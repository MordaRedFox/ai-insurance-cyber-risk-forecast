"""
Главный класс консольного приложения
"""

from cli.menu import Menu
from cli.utils import Console
from cli.commands import build_menu_items


class ConsoleApp:
    """
    Интерактивное меню с командами проекта
    Отвечает только за цикл ввода-вывода и диспетчеризацию команд
    """
    QUIT_KEYS = {'q', 'quit', 'exit', '0'}

    def __init__(self, title: str = 'AI Insurance Cyber Risk Forecast'):
        """
        Args:
            title: заголовок приложения
        """
        self.title = title
        self.menu = Menu('Главное меню', build_menu_items())

    def run(self) -> None:
        """
        Запускает главный цикл приложения
        """
        self._print_banner()
        while True:
            self._print_menu()
            choice = self._prompt()
            if choice in self.QUIT_KEYS:
                Console.info("Выход.")
                return
            item = self.menu.get(choice)
            if item is None or not item.is_actionable:
                Console.error("Неизвестная команда. Попробуйте снова.")
                self._pause()
                continue
            Console.consume_handled()
            try:
                item.action()
            except Exception as exc:
                Console.error(f"Ошибка выполнения: {exc}")
            if not Console.consume_handled():
                self._pause()

    def _print_banner(self) -> None:
        """
        Печатает приветствие
        """
        print(f"\n{Console.BOLD}{Console.CYAN}"
              f"═══ {self.title} ═══{Console.RESET}")
        Console.muted("Прогнозирование киберрисков и андеррайтинг")

    def _print_menu(self) -> None:
        """
        Отображает пункты меню
        """
        print(f"\n{Console.BOLD}{self.menu.title}{Console.RESET}")
        print(f"{'─' * 52}")
        for item in self.menu.items:
            print(f"  {Console.GREEN}{item.key}{Console.RESET}. "
                  f"{Console.BOLD}{item.label}{Console.RESET}")
            if item.description:
                Console.muted(f"     {item.description}")
        print(f"{'─' * 52}")
        Console.muted("  q. Выход")

    @staticmethod
    def _prompt() -> str:
        """
        Читает выбор пользователя
        """
        try:
            return input(f"\n{Console.YELLOW}➜{Console.RESET}  Выбор: ").strip()
        except (EOFError, KeyboardInterrupt):
            return 'q'

    @staticmethod
    def _pause() -> None:
        """
        Приостанавливает выполнение до нажатия Enter
        Дает пользователю время прочитать вывод команды, прежде чем главное
        меню отрисуется снова
        """
        print()
        try:
            input(f"{Console.DIM}Нажмите Enter, чтобы вернуться в меню..."
                  f"{Console.RESET}")
        except (EOFError, KeyboardInterrupt):
            print()
