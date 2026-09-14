"""
Команда препроцессинга данных
"""

from cli.commands.base import BaseCommand
from cli.utils import Console


class PreprocessingCommand(BaseCommand):
    """
    Запускает run_preprocessing.py
    """
    name = 'Препроцессинг данных'
    description = 'Обработать сырые данные: инженерные признаки, кодирование, масштабирование'

    def execute(self) -> None:
        """
        Запускает препроцессинг сырых данных
        """
        Console.title("Препроцессинг данных")
        self.run_script('run_preprocessing.py')
