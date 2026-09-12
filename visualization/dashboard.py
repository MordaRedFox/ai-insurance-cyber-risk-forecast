"""
Оркестратор визуализации
"""

from pathlib import Path
from typing import List, Tuple
from visualization.data_provider import PredictionProvider
from visualization.plots.base import BasePlot
from visualization.plots.probability import ProbabilityPlot
from visualization.plots.damage import DamagePlot
from visualization.plots.scatter import RiskReturnPlot
from visualization.plots.risk_factors import RiskFactorsPlot
from visualization.theme import Theme


class Dashboard:
    """
    Создает все графики презентации
    Управляет только оркестрацией: получает данные, создает графики,
    сохраняет их в указанную папку
    """

    def __init__(self, output_dir: str = 'visualization_output'):
        """
        Args:
            output_dir: директория для сохранения PNG-файлов
        """
        Theme.apply()
        self.data = PredictionProvider().data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def _build_plots(self) -> List[Tuple[str, BasePlot]]:
        """
        Создает список графиков с их именами файлов
        Returns:
            Список графиков
        """
        return [
            ('01_probability', ProbabilityPlot(
                self.data,
                title='Вероятность киберинцидента',
                subtitle='Прогноз на ближайшие 12 месяцев',
            )),
            ('02_damage', DamagePlot(
                self.data,
                title='Прогнозируемый финансовый ущерб',
                subtitle='Ожидаемые потери при инциденте и в среднем',
            )),
            ('03_risk_map', RiskReturnPlot(
                self.data,
                title='Карта рисков',
                subtitle='Связь вероятности инцидента и ожидаемого ущерба',
            )),
            ('04_risk_factors', RiskFactorsPlot(
                self.data,
                title='Ключевые факторы риска',
            )),
        ]

    def render_all(self) -> List[Path]:
        """
        Рендерит все графики и сохраняет их
        Returns:
            Список путей к сохраненным PNG-файлам
        """
        saved: List[Path] = []
        for name, plot in self._build_plots():
            path = self.output_dir / f'{name}.png'
            plot.save(str(path))
            saved.append(path)
            print(f'  ✓ {path}')
        return saved
