from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimulationState:
    episode: int = 0
    step: int = 0
    simulation_status: str = "idle"
    coverage: float = 0.0
    explored_cells: int = 0
    total_cells: int = 0
    collision_count: int = 0
    obstacle_collision_count: int = 0
    reward: float = 0.0
    total_reward: float = 0.0
    wind: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    drones: list[dict] = field(default_factory=list)
    obstacles: list[dict] = field(default_factory=list)
    curriculum_level: int = 1
    visited_cells: list[list[int]] = field(default_factory=list)
    policy_mode: str = "emergent"
    grid_size: int = 20

    def to_dict(self) -> dict:
        return {
            "episode": self.episode,
            "step": self.step,
            "simulation_status": self.simulation_status,
            "coverage": self.coverage,
            "explored_cells": self.explored_cells,
            "total_cells": self.total_cells,
            "collision_count": self.collision_count,
            "obstacle_collision_count": self.obstacle_collision_count,
            "reward": self.reward,
            "total_reward": self.total_reward,
            "wind": list(self.wind),
            "drones": self.drones,
            "obstacles": self.obstacles,
            "curriculum_level": self.curriculum_level,
            "visited_cells": self.visited_cells,
            "policy_mode": self.policy_mode,
            "grid_size": self.grid_size,
        }

