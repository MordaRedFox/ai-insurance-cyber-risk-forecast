"""
Базовый класс для всех команд CLI
"""

from abc import ABC, abstractmethod
from cli.utils import Console, ScriptRunner


class BaseCommand(ABC):
    """
    Абстрактная команда
    Наследники должны реализовать execute() и задать name/description
    """
    name: str = ''
    description: str = ''

    @abstractmethod
    def execute(self) -> None:
        """
        Выполняет команду. Должен быть реализован в наследниках
        """
        raise NotImplementedError

    def run_script(self, script: str, args=None) -> bool:
        """
        Запускает python-скрипт и печатает результат
        Args:
            script: путь к скрипту от корня проекта
            args: список аргументов
        Returns:
            True, если скрипт завершился с кодом 0
        """
        code = ScriptRunner.run(script, args)
        if code == 0:
            Console.success(f"{script} выполнен успешно.")
            return True
        Console.error(f"{script} завершился с кодом {code}.")
        return False
