"""
Пакет коллекции графиков для презентации
"""

from visualization.plots.probability import ProbabilityPlot
from visualization.plots.damage import DamagePlot
from visualization.plots.scatter import RiskReturnPlot
from visualization.plots.risk_factors import RiskFactorsPlot

__all__ = [
    'ProbabilityPlot',
    'DamagePlot',
    'RiskReturnPlot',
    'RiskFactorsPlot',
]
