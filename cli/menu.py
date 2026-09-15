"""
Сущности меню: пункт и список пунктов
"""

from typing import Callable, List, Optional


class MenuItem:
    """
    Один пункт меню
    """

    def __init__(self, key: str, label: str,
                 action: Optional[Callable[[], None]] = None,
                 description: str = ''):
        """
        Args:
            key: символ или строка для выбора (например, "1", "q")
            label: отображаемое название
            action: функция, вызываемая при выборе. None для заголовков
                и разделителей
            description: краткое пояснение под названием
        """
        self.key = key
        self.label = label
        self.action = action
        self.description = description

    @property
    def is_actionable(self) -> bool:
        """
        Является ли пункт выбираемым
        """
        return self.action is not None


class Menu:
    """
    Набор пунктов меню с единым интерфейсом отображения
    """

    def __init__(self, title: str, items: List[MenuItem]):
        """
        Args:
            title: заголовок меню
            items: список пунктов MenuItem
        """
        self.title = title
        self.items = items

    def get(self, key: str) -> Optional[MenuItem]:
        """
        Возвращает пункт по ключу
        """
        for item in self.items:
            if item.key == key:
                return item
        return None
