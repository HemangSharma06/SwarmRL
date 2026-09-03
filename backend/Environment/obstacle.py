from __future__ import annotations

import numpy as np


class DynamicObstacle:
    """Spherical obstacle with deterministic velocity-based movement."""

    def __init__(self, obstacle_id, position, radius=2.0, velocity=None, space_size=100):
        self.obstacle_id = str(obstacle_id)
        self.position = np.asarray(position, dtype=np.float32).copy()
        self.radius = float(radius)
        self.velocity = np.zeros(3, dtype=np.float32) if velocity is None else np.asarray(velocity, dtype=np.float32).copy()
        self.space_size = float(space_size)
        self.active = True

    def move(self, step_size=1.0) -> None:
        if not self.active:
            return
        self.position += self.velocity * float(step_size)
        for axis in range(3):
            if self.position[axis] < 0 or self.position[axis] > self.space_size:
                self.velocity[axis] *= -1
        self.position = np.clip(self.position, 0, self.space_size)

    def collides_with(self, position, radius=0.0) -> bool:
        return self.active and np.linalg.norm(np.asarray(position) - self.position) <= self.radius + radius

    def to_dict(self) -> dict:
        return {
            "id": self.obstacle_id,
            "position": self.position.tolist(),
            "radius": self.radius,
            "velocity": self.velocity.tolist(),
            "active": self.active,
        }
