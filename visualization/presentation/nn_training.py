"""
Анимация обучения многоголовой нейросети
Forward pass (теплые тона) → Loss (пульсация голов) → Backward pass
(холодные тона)
"""

from pathlib import Path
from typing import Sequence, Tuple, List
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.axes import Axes


# Конфигурация
plt.rcParams['font.family'] = 'DejaVu Sans'
BG_COLOR   = '#EAF3FB'
LAYER_FILL = '#5B8FF9'
INPUT_FILL = '#5AD8A6'
HEAD_FILL  = '#F6BD16'
LINE_COLOR = '#7C93A8'
TEXT_COLOR = '#1A2332'
CENTER_X          = 5.75
CIRCLES_PER_LAYER = 7
RADIUS            = 0.18
N_SEGMENTS       = 20
FPS              = 30
DURATION_SECONDS = 9
TOTAL_FRAMES     = FPS * DURATION_SECONDS
N_STAGES         = 5
FORWARD_CMAP  = plt.get_cmap('YlOrRd')
BACKWARD_CMAP = plt.get_cmap('cool')
P_FWD_START,  P_FWD_END  = 0.05, 0.45
P_LOSS_START, P_LOSS_END = 0.45, 0.55
P_BWD_START,  P_BWD_END  = 0.55, 0.95
DEFAULT_OUTPUT = Path('data/visualization_output/presentation/nn_training.gif')
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
                n_shown: int = CIRCLES_PER_LAYER) -> List[Tuple[float, float]]:
    """
    Рисует один слой нейросети в виде кружков и подписи под ним
    Args:
        ax: оси matplotlib, на которых рисуется слой
        x: координата X центра слоя
        count: полное число нейронов в слое (используется только для подписи)
        label: текстовая подпись под слоем
        color: цвет заливки кружков слоя
        n_shown: сколько кружков отображать визуально
    Returns:
        Список координат (x, y) отрисованных кружков слоя
    """
    ys = [1.2 - 2.4 * i / (n_shown - 1) for i in range(n_shown)]
    positions = []
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

def _build_segments(edges: Sequence[Tuple[Tuple[float, float], Tuple[float, float], object, object]],
                    n_segments: int = N_SEGMENTS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Режет ребра на мелкие сегменты для градиентной анимации
    Args:
        edges: список ребер вида (start, end, stage, kind)
        n_segments: число сегментов, на которые делится каждое ребро
    Returns:
        Кортеж (segs, edge_idx_map, params):
            segs - массив сегментов формы (N, 2, 2)
            edge_idx_map - индекс исходного ребра для каждого сегмента
            params - середина параметра t для каждого сегмента
    """
    segs, edge_idx_map, params = [], [], []
    for edge_idx, (start, end, _, _) in enumerate(edges):
        x1, y1 = start
        x2, y2 = end
        for s in range(n_segments):
            t0 = s / n_segments
            t1 = (s + 1) / n_segments
            p1 = (x1 + (x2 - x1) * t0, y1 + (y2 - y1) * t0)
            p2 = (x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1)
            segs.append([p1, p2])
            edge_idx_map.append(edge_idx)
            params.append((t0 + t1) / 2)
    return np.array(segs), np.array(edge_idx_map), np.array(params)

def create_training_animation(output: str | Path = DEFAULT_OUTPUT, fps: int = FPS,
                              duration_seconds: int = DURATION_SECONDS,
                              dpi: int = 150) -> Path:
    """
    Создает и сохраняет GIF-анимацию обучения
    Args:
        output: путь для сохранения GIF
        fps: кадров в секунду
        duration_seconds: длительность анимации в секундах
        dpi: разрешение
    Returns:
        Путь к сохраненному файлу
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    total_frames = fps * duration_seconds
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    layer_positions = []
    for x, count, label, color in LAYERS:
        positions = _draw_layer(ax, x, count, label, color)
        layer_positions.append(positions)
    head_x = 10.8
    for i in range(len(layer_positions) - 1):
        left, right = layer_positions[i], layer_positions[i + 1]
        for j in range(CIRCLES_PER_LAYER):
            for k in range(CIRCLES_PER_LAYER):
                if (j + k) % 2 != 0:
                    continue
                x1, y1 = left[j]
                x2, y2 = right[k]
                ax.plot([x1 + RADIUS, x2 - RADIUS], [y1, y2], color=LINE_COLOR,
                        linewidth=0.35, alpha=0.35, zorder=1)
    last_layer = layer_positions[-1]
    for hx, _, _, _ in HEAD_DATA:
        for lx, ly in last_layer:
            if (int(ly * 10) + int(hx * 10)) % 2 == 0:
                ax.plot([lx + RADIUS, head_x - 0.45], [ly, hx], color=LINE_COLOR,
                        linewidth=0.4, alpha=0.35, zorder=1)
    for hx, hlabel, loss, color in HEAD_DATA:
        rect = patches.FancyBboxPatch(
            (head_x - 0.45, hx - 0.32),
            0.9,
            0.64,
            boxstyle='round,pad=0.05',
            facecolor=color,
            edgecolor='white',
            linewidth=2,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(head_x, hx + 0.05, hlabel, ha='center', va='center', fontsize=10,
                color='white', fontweight='bold', zorder=4)
        ax.text(head_x, hx - 0.42, loss, ha='center', va='top', fontsize=8.5,
                color=TEXT_COLOR, style='italic', zorder=4)
    # Подпись «общий бэкбон»
    ax.annotate('', xy=(2.2, 1.85), xytext=(8.7, 1.85),
                arrowprops=dict(arrowstyle='<->', color=LAYER_FILL, linewidth=1.5))
    ax.text(5.45, 1.95, 'Общий бэкбон', ha='center', va='bottom',
            fontsize=12, color=LAYER_FILL, fontweight='bold')
    # Заголовки по центру холста
    ax.text(CENTER_X, 2.45, 'Архитектура многоголовой нейросети', ha='center',
            va='center', fontsize=15, color=TEXT_COLOR, fontweight='bold')
    ax.text(CENTER_X, 2.20, '32 → 512 → 256 → 128 → 64 → {T1, T2, T3}',
            ha='center', va='center', fontsize=11, color=TEXT_COLOR,
            style='italic')
    # Подпись фазы
    phase_text = ax.text(0.012, 0.98, '', transform=ax.transAxes, ha='left',
                         va='top', fontsize=13, color=TEXT_COLOR,
                         fontweight='bold', zorder=10, bbox=dict(
                             boxstyle='round,pad=0.45',
                             facecolor=BG_COLOR, edgecolor=LINE_COLOR,
                             linewidth=1.0, alpha=0.95))
    ax.set_xlim(-0.3, 11.8)
    ax.set_ylim(-2.1, 2.7)
    ax.axis('off')
    ax.set_aspect('equal')
    plt.tight_layout()
    # Ребра для forward и backward обходов
    all_edges = []
    for i in range(len(layer_positions) - 1):
        left, right = layer_positions[i], layer_positions[i + 1]
        for j in range(CIRCLES_PER_LAYER):
            for k in range(CIRCLES_PER_LAYER):
                all_edges.append((left[j], right[k], i, 'layer'))
    for hx, _, _, _ in HEAD_DATA:
        for j in range(CIRCLES_PER_LAYER):
            all_edges.append((last_layer[j], (head_x - 0.45, hx), 4, 'head'))
    segments, edge_of_seg, seg_param = _build_segments(all_edges)
    lc_fwd = LineCollection(segments, linewidths=1.2, zorder=2,
                            colors=[(0, 0, 0, 0)] * len(segments))
    ax.add_collection(lc_fwd)
    lc_bwd = LineCollection(segments, linewidths=2.0, zorder=2,
                            colors=[(0, 0, 0, 0)] * len(segments))
    ax.add_collection(lc_bwd)

    def _update(frame: int):
        """
        Обновляет кадр анимации: раскрашивает сегменты ребер по текущей фазе
        (forward → loss → backward) и обновляет текст фазы
        Args:
            frame: номер текущего кадра (0 … total_frames - 1)
        Returns:
            Кортеж обновленных артистов (lc_fwd, lc_bwd, phase_text)
            для перерисовки на холсте
        """
        t = frame / total_frames
        fwd_colors = np.zeros((len(segments), 4))
        bwd_colors = np.zeros((len(segments), 4))
        if t < P_FWD_START:
            phase_text.set_text('')
        # Forward: все ребра, включая W1
        elif t < P_FWD_END:
            phase_text.set_text('Forward pass →')
            phase_text.set_color('#C73E1D')
            p = (t - P_FWD_START) / (P_FWD_END - P_FWD_START)
            progress = p * N_STAGES
            for i, (seg_idx, t_local) in enumerate(zip(edge_of_seg, seg_param)):
                stage = all_edges[seg_idx][2]
                hot = progress - stage
                if 0 <= hot <= 1.5:
                    d = abs(t_local - hot)
                    inten = max(0, 1 - d / 0.55) ** 2
                    if inten > 0.01:
                        r, g, b, _ = FORWARD_CMAP(inten)
                        fwd_colors[i] = (r, g, b, inten)
        # Loss: пульсируют ребра, ведущие к головам
        elif t < P_LOSS_END:
            phase_text.set_text('Loss calculation')
            phase_text.set_color('#C73E1D')
            p = (t - P_LOSS_START) / (P_LOSS_END - P_LOSS_START)
            pulse = np.sin(p * np.pi)
            for i, seg_idx in enumerate(edge_of_seg):
                if all_edges[seg_idx][3] == 'head':
                    r, g, b, _ = FORWARD_CMAP(pulse)
                    fwd_colors[i] = (r, g, b, pulse)
        # Backward: тоже все ребра, включая W1
        elif t < P_BWD_END:
            phase_text.set_text('← Backward pass')
            phase_text.set_color('#5B8FF9')
            p = (t - P_BWD_START) / (P_BWD_END - P_BWD_START)
            progress = p * N_STAGES
            for i, (seg_idx, t_local) in enumerate(zip(edge_of_seg, seg_param)):
                stage = all_edges[seg_idx][2]
                stage_back = (N_STAGES - 1) - stage
                hot = progress - stage_back
                if 0 <= hot <= 1.5:
                    d = abs((1 - t_local) - hot)
                    inten = max(0, 1 - d / 0.55) ** 2
                    if inten > 0.01:
                        r, g, b, _ = BACKWARD_CMAP(inten)
                        bwd_colors[i] = (r, g, b, inten)
        else:
            phase_text.set_text('')
        lc_fwd.set_color(fwd_colors)
        lc_bwd.set_color(bwd_colors)
        return lc_fwd, lc_bwd, phase_text

    anim = FuncAnimation(fig, _update, frames=total_frames, interval=1000 / fps,
                         blit=False)
    anim.save(output, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"Сохранено в файл: {output}")
    return output


if __name__ == '__main__':
    create_training_animation()
