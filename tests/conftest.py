"""Общие настройки pytest.

Добавляем корень проекта в sys.path, чтобы тесты могли импортировать
пакеты common/, most_popular/, fpmc/, light_gcn/, sasrec/, train/.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))