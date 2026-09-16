"""
Команды визуализации и открытия графиков
"""

from pathlib import Path
from cli.commands.base import BaseCommand
from cli.services import FileFinder, FolderOpener
from cli.utils import Console


OUTPUT_DIR = Path('data') / 'visualization_output'


class VisualizeCommand(BaseCommand):
    """
    Запускает run_visualize.py
    """
    name = 'Создать графики'
    description = 'Построить графики по последнему инференсу'

    def execute(self) -> None:
        """
        Запускает генерацию графиков по данным инференса
        """
        Console.title("Генерация графиков")
        self.run_script('run_scripts/run_visualize.py')


class OpenVisualizationsCommand(BaseCommand):
    """
    Открывает папку с графиками
    """
    name = 'Открыть графики'
    description = 'Показать созданные PNG-файлы в системном просмотрщике'

    def __init__(self):
        """
        Инициализирует сервисы для поиска файлов и открытия папок
        """
        self.file_finder = FileFinder()
        self.folder_opener = FolderOpener()

    def execute(self) -> None:
        """
        Открывает папку с сохраненными PNG-графиками
        """
        Console.title("Открытие визуализаций")
        if not OUTPUT_DIR.exists():
            Console.error(f"Папка {OUTPUT_DIR} не найдена. "
                          "Сначала создайте визуализации.")
            return
        files = self.file_finder.find(OUTPUT_DIR, '*.png')
        if not files:
            Console.error("Файлы не найдены.")
            return
        Console.info(f"Найдено {len(files)} файлов:")
        for f in files:
            Console.muted(f"  • {f.name}")
        try:
            self.folder_opener.open(OUTPUT_DIR)
            Console.success("Папка открыта.")
        except RuntimeError as exc:
            Console.error(str(exc))
