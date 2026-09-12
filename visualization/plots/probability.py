"""
График вероятности инцидента (T1)
"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from visualization.plots.base import BasePlot
from visualization.theme import Theme


class ProbabilityPlot(BasePlot):
    """
    Горизонтальная диаграмма вероятностей инцидента
    """

    def render(self) -> Figure:
        """
        Создает горизонтальную диаграмму вероятности инцидента
        Returns:
            Горизонтальная диаграмма
        """
        df = self.data.sort_values('T1', ascending=True)
        fig, ax = plt.subplots(figsize=Theme.FIGSIZE_SINGLE)
        labels = [
            self.format_company_label(row['Отрасль'],
                                      row['Годовая выручка (млн руб.)'])
            for _, row in df.iterrows()
        ]
        values = df['T1'].values
        colors = [Theme.risk_color(v) for v in values]
        bars = ax.barh(labels, values, color=colors, height=0.62,
                       edgecolor=Theme.COLORS['white'], linewidth=1.5)
        for bar, val in zip(bars, values):
            ax.text(
                val - 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:.1%}', va='center', ha='right',
                fontsize=Theme.FONTS['value'], fontweight='bold',
                color=Theme.COLORS['white'],
            )
        ax.set_xlim(0, 1.0)
        ax.set_xlabel('Вероятность инцидента')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        self._add_title(ax)
        fig.tight_layout()
        return fig
