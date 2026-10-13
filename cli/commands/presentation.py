"""
Команды создания и открытия визуализаций для презентации нейросети
"""

from pathlib import Path
from cli.commands.base import BaseCommand
from cli.services import FileFinder, FolderOpener
from cli.utils import Console


OUTPUT_DIR = Path('data') / 'visualization_output' / 'presentation'


class PresentationCommand(BaseCommand):
    """
    Запускает run_presentation.py
    """
    name = 'Создать визуализации для презентации'
    description = 'Построить схему архитектуры и анимацию обучения нейросети'

    def execute(self) -> None:
        """
        Запускает генерацию визуализаций для презентации
        """
        Console.title("Генерация презентационных визуализаций")
        self.run_script('run_scripts/run_presentation.py')


class OpenPresentationCommand(BaseCommand):
    """
    Открывает папку с визуализациями для презентации
    """
    name = 'Открыть визуализации для презентации'
    description = 'Показать созданные файлы в системном просмотрщике'

    def __init__(self):
        """
        Инициализирует сервисы для поиска файлов и открытия папок
        """
        self.file_finder = FileFinder()
        self.folder_opener = FolderOpener()

    def execute(self) -> None:
        """
        Открывает папку с визуализациями для презентации
        """
        Console.title("Открытие презентационных визуализаций")
        if not OUTPUT_DIR.exists():
            Console.error(f"Папка {OUTPUT_DIR} не найдена. "
                          "Сначала создайте визуализации.")
            return
        files = (self.file_finder.find(OUTPUT_DIR, '*.png')
                 + self.file_finder.find(OUTPUT_DIR, '*.gif'))
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
