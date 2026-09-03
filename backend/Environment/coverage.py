from __future__ import annotations

from typing import Iterable

import numpy as np


class CoverageTracker:
    """Track visited cells in the cubic search space."""

    def __init__(self, space_size: float, grid_size: int = 20):
        if grid_size < 1:
            raise ValueError("grid_size must be positive")
        self.space_size = float(space_size)
        self.grid_size = int(grid_size)
        self.visited_cells: set[tuple[int, int, int]] = set()

    @property
    def total_cells(self) -> int:
        return self.grid_size ** 3

    @property
    def explored_cells(self) -> int:
        return len(self.visited_cells)

    @property
    def coverage_percentage(self) -> float:
        return 100.0 * self.explored_cells / self.total_cells

    def _cell_for_position(self, position: Iterable[float]) -> tuple[int, int, int]:
        coordinates = np.asarray(list(position), dtype=np.float32)
        indices = np.floor(coordinates / self.space_size * self.grid_size).astype(int)
        return tuple(np.clip(indices, 0, self.grid_size - 1).tolist())

    def visit(self, position: Iterable[float]) -> bool:
        cell = self._cell_for_position(position)
        was_new = cell not in self.visited_cells
        self.visited_cells.add(cell)
        return was_new

    def visit_positions(self, positions: Iterable[Iterable[float]]) -> int:
        return sum(self.visit(position) for position in positions)

    def reset(self) -> None:
        self.visited_cells.clear()

    def to_dict(self) -> dict:
        return {
            "visited_cells": [list(cell) for cell in sorted(self.visited_cells)],
            "total_cells": self.total_cells,
            "explored_cells": self.explored_cells,
            "coverage_percentage": self.coverage_percentage,
        }
