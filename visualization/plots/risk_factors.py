"""
График ключевых факторов риска для наиболее рискованной компании
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, List
from matplotlib.figure import Figure
from visualization.plots.base import BasePlot
from visualization.theme import Theme


class RiskFactorsPlot(BasePlot):
    """
    Горизонтальная диаграмма нормализованных факторов риска
    """
    FACTORS = [
        ('Объем персональных данных',
         'Объем обрабатываемых персональных данных (количество записей, млн)',
         100, False),
        ('Количество IoT/OT-устройств',
         'Количество подключенных IoT/OT-устройств', 10000, False),
        ('Устаревшее ПО (EOL)',
         'Наличие устаревшего ПО/EOL', 1, False),
        ('Отсутствие сегментации сети',
         'Наличие сегментации сети', 1, True),
        ('Низкий % обученных сотрудников',
         'Процент сотрудников, прошедших обучение кибергигиене за год',
         1, True),
        ('Редкий пентест',
         'Периодичность внешнего пентеста', 24, False),
    ]

    def render(self) -> Figure:
        """
        Создает горизонтальную диаграмму нормализованных факторов риска
        Returns:
            Горизонтальная диаграмма
        """
        company = self.data.loc[self.data['T1'].idxmax()]
        names, values = self._compute(company)
        order = np.argsort(values)
        names = [names[i] for i in order]
        values = [values[i] for i in order]
        fig, ax = plt.subplots(figsize=Theme.FIGSIZE_SINGLE)
        colors = [Theme.risk_color(v) for v in values]
        bars = ax.barh(names, values, color=colors, height=0.65,
                       edgecolor=Theme.COLORS['white'], linewidth=1.2)
        for bar, val in zip(bars, values):
            ax.text(
                val + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:.0%}', va='center', ha='left',
                fontsize=Theme.FONTS['value'], fontweight='bold',
                color=Theme.COLORS['text'],
            )
        ax.set_xlim(0, 1.05)
        ax.set_xlabel('Уровень риска')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        industry = company['Отрасль'].capitalize()
        revenue = company['Годовая выручка (млн руб.)']
        self.subtitle = (
            f'Компания с максимальным риском: '
            f'{industry} ({revenue:.0f} млн ₽)'
        )
        self._add_title(ax)
        fig.tight_layout()
        return fig

    def _compute(self, company: pd.Series) -> Tuple[List[str], List[float]]:
        """
        Вычисляет нормализованные значения факторов
        Args:
        company: строка DataFrame (pd.Series) с данными одной компании,
            содержащая поля, указанные в self.FACTORS
        Returns:
            Кортеж из двух списков:
                - names: названия факторов риска (для подписей оси Y)
                - values: нормализованные значения риска в диапазоне [0, 1]
        """
        names, values = [], []
        for name, field, max_val, invert in self.FACTORS:
            if field not in company:
                continue
            raw = float(company[field])
            normalized = min(raw / max_val, 1.0)
            if invert:
                normalized = 1.0 - normalized
            names.append(name)
            values.append(normalized)
        return names, values
