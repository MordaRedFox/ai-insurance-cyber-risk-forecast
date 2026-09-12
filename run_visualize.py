"""
Скрипт генерации визуализаций для презентации
"""

import sys
sys.path.append('.')
from visualization import Dashboard


if __name__ == '__main__':
    print("Генерация графиков визуализации...")
    dashboard = Dashboard(output_dir='data/visualization_output')
    saved = dashboard.render_all()
    print(f"\nГотово. Сохранено {len(saved)} файлов")
