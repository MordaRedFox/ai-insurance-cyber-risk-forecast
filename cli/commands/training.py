"""
Команда обучения модели
"""

from cli.commands.base import BaseCommand
from cli.utils import Console


class TrainingCommand(BaseCommand):
    """
    Запускает run_training.py
    """
    name = 'Обучение модели'
    description = 'Обучить многоголовую нейросеть на обработанных данных'

    def execute(self) -> None:
        """
        Запускает обучение модели на данных после препроцессинга
        """
        Console.title("Обучение модели")
        Console.warning("Это может занять продолжительное время.")
        self.run_script('run_scripts/run_training.py')
