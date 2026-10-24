"""
Словарь переводов идентификаторов для отображения на графиках
"""

INDUSTRY_LABELS = {
    'finance':       'Финансы',
    'retail':        'Ритейл',
    'it_telecom':    'ИТ/Телеком',
    'manufacturing': 'Производство',
    'healthcare':    'Здравоохранение',
    'other':         'Другое',
}

def industry_label(code: str) -> str:
    """
    Переводит идентификатор отрасли в русское название
    Args:
        code: код отрасли из датасета ('finance', 'retail', ...)
    Returns:
        Русское название. Если код незнаком, возвращается сам код
        с заглавной буквы
    """
    return INDUSTRY_LABELS.get(code, code.capitalize())
