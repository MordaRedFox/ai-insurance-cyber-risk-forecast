"""
График ожидаемого ущерба (T2 и T3)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from visualization.plots.base import BasePlot
from visualization.theme import Theme


class DamagePlot(BasePlot):
    """
    Парная диаграмма условного и безусловного ущерба
    """

    def render(self) -> Figure:
        """
        Создает парную диаграмму условного и безусловного ущерба
        Returns:
            Парная диаграмма
        """
        df = self.data.copy()
        df['damage_t2'] = np.exp(df['T2']) - 1
        df['damage_t3'] = np.exp(df['T3']) - 1
        df = df.sort_values('damage_t3', ascending=False)
        fig, ax = plt.subplots(figsize=Theme.FIGSIZE_WIDE)
        labels = [
            self.format_company_label(row['Отрасль'],
                                      row['Годовая выручка (млн руб.)'])
            for _, row in df.iterrows()
        ]
        x = np.arange(len(df))
        width = 0.38
        bars1 = ax.bar(
            x - width / 2, df['damage_t2'], width,
            label='Условный ущерб (если инцидент случится)',
            color=Theme.COLORS['primary'],
            edgecolor=Theme.COLORS['white'], linewidth=1.2,
        )
        bars2 = ax.bar(
            x + width / 2, df['damage_t3'], width,
            label='Безусловный ущерб (в среднем)',
            color=Theme.COLORS['orange'],
            edgecolor=Theme.COLORS['white'], linewidth=1.2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=Theme.FONTS['tick'])
        ax.set_yscale('log')
        ax.set_ylabel('Ожидаемый ущерб, млн ₽ (лог. шкала)')
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: self.format_money(v, short=True)))
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', frameon=False)
        for bars, values in [(bars1, df['damage_t2']), (bars2, df['damage_t3'])]:
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, val * 1.15,
                    self.format_money(val, short=True),
                    ha='center', va='bottom',
                    fontsize=Theme.FONTS['value'] - 1,
                    color=Theme.COLORS['text'],
                )
        self._add_title(ax)
        fig.tight_layout()
        return fig
