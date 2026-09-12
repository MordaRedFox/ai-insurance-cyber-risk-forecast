"""
Карта рисков: вероятность vs ожидаемый ущерб
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from visualization.plots.base import BasePlot
from visualization.theme import Theme


class RiskReturnPlot(BasePlot):
    """
    Диаграмма рассеяния: вероятность инцидента vs ущерб
    """
    # Палитра для отраслей
    INDUSTRY_COLORS = {
        'finance': '#5B8FF9',
        'retail': '#5AD8A6',
        'it_telecom': '#9270CA',
        'manufacturing': '#E8684A',
        'healthcare': '#F6BD16',
        'other': '#95A5A6',
    }

    def render(self) -> Figure:
        """
        Создает диаграмму рассеяния
        Returns:
            Диаграмма рассеяния
        """
        df = self.data.copy()
        df['damage'] = np.exp(df['T3']) - 1
        fig, ax = plt.subplots(figsize=Theme.FIGSIZE_SQUARE)
        max_rev = df['Годовая выручка (млн руб.)'].max()
        sizes = 120 + (df['Годовая выручка (млн руб.)'] / max_rev) * 900
        for industry, color in self.INDUSTRY_COLORS.items():
            mask = df['Отрасль'] == industry
            if not mask.any():
                continue
            ax.scatter(
                df.loc[mask, 'T1'], df.loc[mask, 'damage'],
                s=sizes[mask], alpha=0.85,
                color=color, edgecolor=Theme.COLORS['white'],
                linewidth=2, label=industry.capitalize(), zorder=3,
            )
        for _, row in df.iterrows():
            ind = row['Отрасль'].capitalize()
            revenue = row['Годовая выручка (млн руб.)']
            ax.annotate(
                f"{ind} · {revenue:.0f}",
                (row['T1'], row['damage']),
                xytext=(9, 9), textcoords='offset points',
                fontsize=Theme.FONTS['tick'] - 1,
                color=Theme.COLORS['text'],
            )
        ax.set_xlabel('Вероятность инцидента')
        ax.set_ylabel('Ожидаемый ущерб, млн ₽ (лог. шкала)')
        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        ax.legend(
            loc='lower right', frameon=False,
            title='Отрасль', title_fontsize=Theme.FONTS['label'],
        )
        self._add_title(ax)
        fig.tight_layout()
        return fig
