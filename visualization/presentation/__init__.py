"""
Пакет для визуализации архитектуры и обучения нейросети
"""

from visualization.presentation.nn_architectural import create_plot_architecture
from visualization.presentation.nn_training import create_training_animation

__all__ = [
    "create_plot_architecture",
    "create_training_animation",
]
