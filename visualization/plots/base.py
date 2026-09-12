"""
Абстрактный базовый класс для всех графиков
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from visualization.theme import Theme


class BasePlot(ABC):
    """
    Базовый класс для графиков
    Обеспечивает единый интерфейс render/save и общие утилиты
    """

    def __init__(self, data: pd.DataFrame, title: str = '', subtitle: str = ''):
        """
        Args:
            data: DataFrame с предсказаниями
            title: основной заголовок
            subtitle: подзаголовок
        """
        self.data = data
        self.title = title
        self.subtitle = subtitle
        self._figure: Optional[Figure] = None

    @abstractmethod
    def render(self) -> Figure:
        """
        Создает фигуру. Должен быть реализован в наследниках
        """
        raise NotImplementedError

    def save(self, path: str) -> Path:
        """
        Сохраняет график в файл
        Args:
            path: путь для сохранения
        Returns:
            Путь к сохраненному файлу
        """
        if self._figure is None:
            self._figure = self.render()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._figure.savefig(
            path, dpi=Theme.DPI, bbox_inches='tight',
            facecolor=self._figure.get_facecolor(),
        )
        plt.close(self._figure)
        return path

    def _add_title(self, ax) -> None:
        """
        Добавляет заголовок и подзаголовок
        """
        if self.title:
            ax.set_title(
                self.title, fontsize=Theme.FONTS['title'],
                fontweight='bold', color=Theme.COLORS['dark'],
                loc='left', pad=22 if self.subtitle else 10,
            )
        if self.subtitle:
            ax.text(
                0, 1.02, self.subtitle,
                transform=ax.transAxes,
                fontsize=Theme.FONTS['subtitle'],
                color=Theme.COLORS['text_muted'],
                ha='left', va='bottom',
            )

    @staticmethod
    def format_company_label(industry: str, revenue: float) -> str:
        """
        Форматирует метку компании для оси
        Args:
            industry: название отрасли
            revenue: годовая выручка в млн руб
        Returns:
            Строка вида "Finance\\n350 млн ₽"
        """
        name = industry.capitalize()
        if revenue >= 1000:
            rev = f'{revenue / 1000:.1f} млрд ₽'
        else:
            rev = f'{revenue:.0f} млн ₽'
        return f'{name}\n{rev}'

    @staticmethod
    def format_money(value: float, short: bool = False) -> str:
        """
        Форматирует денежные значения
        Args:
            value: сумма в млн руб
            short: если True - компактный формат (K/M)
        Returns:
            Отформатированная строка
        """
        if value >= 1_000_000:
            return f'{value / 1_000_000:.1f}M'
        if value >= 1000:
            return f'{value / 1000:.1f}K' if short else f'{value:,.0f} млн'
        return f'{value:.0f}' if short else f'{value:.0f} млн'
