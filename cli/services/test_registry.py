"""
Реестр тестовых модулей проекта
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class TestModule:
    """
    Описание одного тестового модуля
    """
    key: str
    path: str
    label: str


class TestRegistry:
    """
    Хранит список доступных тестовых модулей
    """
    MODULES: List[TestModule] = [
        TestModule('1', 'tests/test_data_generation.py', 'Генерация данных'),
        TestModule('2', 'tests/test_preprocessing.py', 'Препроцессинг'),
        TestModule('3', 'tests/test_model.py', 'Модель'),
        TestModule('4', 'tests/test_train.py', 'Обучение'),
        TestModule('5', 'tests/test_infer.py', 'Инференс'),
    ]

    def all(self) -> List[TestModule]:
        """
        Возвращает все зарегистрированные модули
        Returns:
            Список тестовых модулей
        """
        return list(self.MODULES)

    def find(self, key: str) -> Optional[TestModule]:
        """
        Находит модуль по ключу
        Args:
            key: ключ модуля
        Returns:
            Найденный модуль или None
        """
        for module in self.MODULES:
            if module.key == key:
                return module
        return None
