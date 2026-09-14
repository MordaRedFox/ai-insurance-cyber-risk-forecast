"""
Команда генерации синтетических данных
"""

from cli.commands.base import BaseCommand
from cli.utils import Console


class GenerationCommand(BaseCommand):
    """
    Запускает run_generation.py
    """
    name = 'Генерация данных'
    description = 'Создать синтетический датасет компаний (data/raw/)'

    def execute(self) -> None:
        """
        Запускает генерацию синтетических данных
        """
        Console.title("Генерация синтетических данных")
        self.run_script('run_generation.py')
