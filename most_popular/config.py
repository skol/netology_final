"""Настройки рекомендательной системы MostPopular для VK-LSVD.

Здесь вынесены параметры тренировочного периода: начальная и конечная неделя.
Валидационная неделя всегда следует сразу за тренировочной частью (end_week + 1)
без разрывов.
"""

from dataclasses import dataclass


@dataclass
class MostPopularConfig:
    # Путь к подвыборке датасета VK-LSVD.
    data_dir: str = "data/raw/VK-LSVD/subsamples/up0.001_ip0.001"

    # Тренировочный период: недели [start_week, end_week] включительно.
    start_week: int = 0
    end_week: int = 23

    # Порог времени просмотра для положительного примера (timespent >= 5).
    min_timespent_pos: int = 5

    # Глубина топ-k для метрик и рекомендаций.
    k: int = 10

    @property
    def val_week(self) -> int:
        """Валидационная неделя идёт сразу за тренировочной частью."""
        return self.end_week + 1

    @property
    def train_weeks(self) -> range:
        return range(self.start_week, self.end_week + 1)