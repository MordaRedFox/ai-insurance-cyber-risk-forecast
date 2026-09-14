"""
Пакт реестра всех команд CLI
"""

from typing import List
from cli.commands.generation import GenerationCommand
from cli.commands.preprocessing import PreprocessingCommand
from cli.commands.training import TrainingCommand
from cli.commands.inference import InferenceCommand
from cli.commands.visualization import (
    VisualizeCommand, OpenVisualizationsCommand)
from cli.commands.tests import RunAllTestsCommand, RunTestModuleCommand
from cli.menu import MenuItem


def build_menu_items() -> List[MenuItem]:
    """
    Создает список MenuItem для главного меню
    Returns:
        Список пунктов меню в порядке отображения
    """
    return [
        MenuItem('1', GenerationCommand.name,
                 GenerationCommand().execute,
                 GenerationCommand.description),
        MenuItem('2', PreprocessingCommand.name,
                 PreprocessingCommand().execute,
                 PreprocessingCommand.description),
        MenuItem('3', TrainingCommand.name,
                 TrainingCommand().execute,
                 TrainingCommand.description),
        MenuItem('4', InferenceCommand.name,
                 InferenceCommand().execute,
                 InferenceCommand.description),
        MenuItem('5', VisualizeCommand.name,
                 VisualizeCommand().execute,
                 VisualizeCommand.description),
        MenuItem('6', OpenVisualizationsCommand.name,
                 OpenVisualizationsCommand().execute,
                 OpenVisualizationsCommand.description),
        MenuItem('7', RunAllTestsCommand.name,
                 RunAllTestsCommand().execute,
                 RunAllTestsCommand.description),
        MenuItem('8', RunTestModuleCommand.name,
                 RunTestModuleCommand().execute,
                 RunTestModuleCommand.description),
    ]
