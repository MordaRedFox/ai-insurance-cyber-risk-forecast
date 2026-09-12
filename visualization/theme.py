"""
Единая тема оформления для всех графиков
Содержит палитру, размеры шрифтов, метод применения стиля
"""

import matplotlib.pyplot as plt
from typing import Dict, Type


class Theme:
    """
    Централизованное управление стилем визуализации
    """
    # Цветовая палитра
    COLORS: Dict[str, str] = {
        'primary': '#5B8FF9',
        'secondary': '#5AD8A6',
        'accent': '#F6BD16',
        'danger': '#E8684A',
        'light_blue': '#6DC8EC',
        'purple': '#9270CA',
        'orange': '#FF9D4D',
        'teal': '#269A99',
        'neutral': '#95A5A6',
        'dark': '#1A2332',
        'bg': '#FFFFFF',
        'card_bg': '#FAFBFC',
        'grid': '#E8ECEF',
        'text': '#2C3E50',
        'text_muted': '#8592A0',
        'white': '#FFFFFF',
    }
    # Размеры шрифтов (в пунктах)
    FONTS = {
        'title': 15,
        'subtitle': 11,
        'label': 10,
        'tick': 10,
        'value': 10,
    }
    # Размеры фигур (в дюймах)
    FIGSIZE_SINGLE = (10, 5.5)
    FIGSIZE_WIDE = (11, 5.5)
    FIGSIZE_SQUARE = (9, 6)
    DPI = 300
    # Пороги для окрашивания риска
    RISK_LOW = 0.5
    RISK_MEDIUM = 0.75

    @classmethod
    def apply(cls: Type['Theme']) -> None:
        """
        Применяет единый стиль ко всем графикам matplotlib
        """
        plt.rcParams.update({
            'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
            'font.size': cls.FONTS['label'],
            'axes.titlesize': cls.FONTS['title'],
            'axes.titleweight': 'bold',
            'axes.titlecolor': cls.COLORS['dark'],
            'axes.labelsize': cls.FONTS['label'],
            'axes.labelcolor': cls.COLORS['text'],
            'axes.edgecolor': cls.COLORS['grid'],
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.facecolor': cls.COLORS['bg'],
            'xtick.color': cls.COLORS['text_muted'],
            'ytick.color': cls.COLORS['text_muted'],
            'xtick.labelsize': cls.FONTS['tick'],
            'ytick.labelsize': cls.FONTS['tick'],
            'grid.color': cls.COLORS['grid'],
            'grid.linewidth': 0.8,
            'grid.alpha': 0.6,
            'legend.frameon': False,
            'legend.fontsize': cls.FONTS['label'],
            'figure.facecolor': cls.COLORS['bg'],
            'figure.dpi': cls.DPI,
            'savefig.dpi': cls.DPI,
            'savefig.bbox': 'tight',
            'savefig.facecolor': cls.COLORS['bg'],
            'savefig.pad_inches': 0.3,
        })

    @classmethod
    def risk_color(cls: Type['Theme'], value: float) -> str:
        """
        Возвращает цвет в зависимости от уровня риска
        Args:
            value: значение риска в диапазоне [0, 1]
        Returns:
            HEX-код цвета
        """
        if value < cls.RISK_LOW:
            return cls.COLORS['secondary']
        if value < cls.RISK_MEDIUM:
            return cls.COLORS['accent']
        return cls.COLORS['danger']
