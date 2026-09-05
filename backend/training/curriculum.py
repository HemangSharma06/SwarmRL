from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CurriculumLevel:
    level: int
    obstacle_count: int
    dynamic_obstacles: bool
    wind_strength: float


class CurriculumManager:
    """Deterministic difficulty schedule driven by completed training episodes."""

    def __init__(self, enabled=True, levels=None):
        self.enabled = bool(enabled)
        self.levels = levels or [
            CurriculumLevel(1, 0, False, 0.0),
            CurriculumLevel(2, 2, False, 0.15),
            CurriculumLevel(3, 3, True, 0.35),
            CurriculumLevel(4, 5, True, 0.6),
        ]
        self.episode_count = 0

    @property
    def current_level(self) -> CurriculumLevel:
        if not self.enabled:
            return self.levels[0]
        return self.levels[min(self.episode_count // 100, len(self.levels) - 1)]

    def advance(self, completed_episodes=1) -> CurriculumLevel:
        self.episode_count += int(completed_episodes)
        return self.current_level

    def reset(self) -> None:
        self.episode_count = 0

    def to_dict(self) -> dict:
        level = self.current_level
        return {
            "level": level.level,
            "episode_count": self.episode_count,
            "obstacle_count": level.obstacle_count,
            "dynamic_obstacles": level.dynamic_obstacles,
            "wind_strength": level.wind_strength,
        }
