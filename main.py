"""
Точка входа консольного приложения (скрипт запуска)
"""

import sys
sys.path.append('.')
from cli import ConsoleApp


if __name__ == '__main__':
    app = ConsoleApp()
    app.run()
