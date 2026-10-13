"""
Запуск генерации визуализаций для презентации нейросети
Создает схему архитектуры и анимацию обучения
"""

from pathlib import Path
import sys
sys.path.append('.')
from visualization import create_plot_architecture, create_training_animation


PRESENTATION_DIR = Path('data') / 'visualization_output' / 'presentation'


if __name__ == '__main__':
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    create_plot_architecture(PRESENTATION_DIR / 'nn_architecture.png')
    create_training_animation(PRESENTATION_DIR / 'nn_training.gif')
