"""
Команда инференса модели
"""

from cli.commands.base import BaseCommand
from cli.utils import Console


class InferenceCommand(BaseCommand):
    """
    Запускает run_inference.py
    """
    name = 'Инференс модели'
    description = 'Сделать прогноз для презентационного датасета'

    def execute(self) -> None:
        """
        Запускает инференс модели
        """
        Console.title("Инференс модели")
        self.run_script('run_scripts/run_inference.py')
