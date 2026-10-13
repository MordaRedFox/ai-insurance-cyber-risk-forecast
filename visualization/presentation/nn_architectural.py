"""
Визуализация архитектуры нейросети для презентации
"""

from pathlib import Path
from typing import List, Tuple
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


# Конфигурация
plt.rcParams['font.family'] = 'DejaVu Sans'
BG_COLOR     = '#EAF3FB'
LAYER_FILL   = '#5B8FF9'
INPUT_FILL   = '#5AD8A6'
HEAD_FILL    = '#F6BD16'
LINE_COLOR   = '#7C93A8'
TEXT_COLOR   = '#1A2332'
CENTER_X = 5.75
CIRCLES_PER_LAYER = 7
RADIUS = 0.18
DEFAULT_OUTPUT = Path('data/visualization_output/presentation/nn_architecture.png')
LAYERS = [
    (0.5, 32,  'Вход\n(32)',         INPUT_FILL),
    (2.5, 512, 'Скрытый 1\n(512)',   LAYER_FILL),
    (4.5, 256, 'Скрытый 2\n(256)',   LAYER_FILL),
    (6.5, 128, 'Скрытый 3\n(128)',   LAYER_FILL),
    (8.5, 64,  'Признаки\n(64)',     LAYER_FILL),
]
HEAD_DATA = [
    (1.0,  'T1\nВероятность\nинцидента', 'Sigmoid + BCE', HEAD_FILL),
    (0.0,  'T2\nУсловный\nущерб',        'MSE',           HEAD_FILL),
    (-1.0, 'T3\nБезусловный\nущерб',     'MSE',           HEAD_FILL),
]

def _draw_layer(ax: Axes, x: float, count: int, label: str, color: str,
               n_shown: int = CIRCLES_PER_LAYER,) -> List[Tuple[float, float]]:
    """
    Рисует один слой нейросети в виде набора кружков и подписи под ним
    Args:
        ax: оси matplotlib
        x: координата X центра слоя
        count: полное число нейронов в слое, используется только для подписи
        label: текстовая подпись под слоем
        color: цвет заливки кружков слоя
        n_shown: сколько кружков отображать визуально
    Returns:
        Список координат (x, y) отрисованных кружков слоя
    """
    positions = []
    ys = [1.2 - 2.4 * i / (n_shown - 1) for i in range(n_shown)]
    for i, y in enumerate(ys):
        if i == n_shown // 2:
            positions.append((x, y))
            continue
        circle = patches.Circle((x, y), RADIUS, facecolor=color,
                                edgecolor='white', linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        positions.append((x, y))
    ax.text(x, -1.75, label, ha='center', va='top', fontsize=11,
            color=TEXT_COLOR, fontweight='bold')
    ax.text(x, 0, '⋮', ha='center', va='center', fontsize=22,
            color=TEXT_COLOR, zorder=4)
    return positions

def _draw_architecture(ax: Axes) -> None:
    """
    Рисует всю архитектуру на переданных осях
    """
    layer_positions = []
    for x, count, label, color in LAYERS:
        positions = _draw_layer(ax, x, count, label, color)
        layer_positions.append(positions)
    # Связи между слоями
    for i in range(len(layer_positions) - 1):
        left = layer_positions[i]
        right = layer_positions[i + 1]
        for j in range(CIRCLES_PER_LAYER):
            for k in range(CIRCLES_PER_LAYER):
                if (j + k) % 2 != 0:
                    continue
                x1, y1 = left[j]
                x2, y2 = right[k]
                ax.plot([x1 + RADIUS, x2 - RADIUS], [y1, y2], color=LINE_COLOR,
                        linewidth=0.35, alpha=0.5, zorder=1)
    head_x = 10.8
    last_layer = layer_positions[-1]
    for hx, hlabel, _, _ in HEAD_DATA:
        for lx, ly in last_layer:
            if (int(ly * 10) + int(hx * 10)) % 2 == 0:
                ax.plot([lx + RADIUS, head_x - 0.45], [ly, hx], color=LINE_COLOR,
                        linewidth=0.4, alpha=0.45, zorder=1)
    for hx, hlabel, loss, color in HEAD_DATA:
        rect = patches.FancyBboxPatch(
            (head_x - 0.45, hx - 0.32),
            0.9,
            0.64,
            boxstyle='round,pad=0.05',
            facecolor=color,
            edgecolor='white',
            linewidth=2,
            zorder=3
        )
        ax.add_patch(rect)
        ax.text(head_x, hx + 0.05, hlabel, ha='center', va='center',
                fontsize=10, color='white', fontweight='bold', zorder=4)
        ax.text(head_x, hx - 0.42, loss, ha='center', va='top', fontsize=8.5,
                color=TEXT_COLOR, style='italic',zorder=4)
    # Подпись «общий бэкбон»
    ax.annotate('', xy=(2.2, 1.85), xytext=(8.7, 1.85),
                arrowprops=dict(arrowstyle='<->', color=LAYER_FILL, linewidth=1.5))
    ax.text(5.45, 1.95, 'Общий бэкбон', ha='center', va='bottom', fontsize=12,
            color=LAYER_FILL, fontweight='bold')
    # Заголовки по центру холста
    ax.text(CENTER_X, 2.45, 'Архитектура многоголовой нейросети', ha='center',
            va='center', fontsize=15, color=TEXT_COLOR, fontweight='bold')
    ax.text(CENTER_X, 2.20, '32 → 512 → 256 → 128 → 64 → {T1, T2, T3}',
            ha='center', va='center', fontsize=11, color=TEXT_COLOR, style='italic')
    ax.set_xlim(-0.3, 11.8)
    ax.set_ylim(-2.1, 2.7)
    ax.axis('off')
    ax.set_aspect('equal')

def create_plot_architecture(output: str | Path = DEFAULT_OUTPUT, dpi: int = 300,
                      show: bool = False) -> Path:
    """
    Строит и сохраняет схему архитектуры
    Args:
        output: путь для сохранения PNG
        dpi: разрешение
        show: показывать ли окно matplotlib
    Returns:
        Путь к сохраненному файлу
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    _draw_architecture(ax)
    plt.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches='tight', facecolor=BG_COLOR)
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f'Сохранено в файл: {output}')
    return output


if __name__ == '__main__':
    create_plot_architecture()
