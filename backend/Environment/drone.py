import numpy as np


class Drone:

    def __init__(self, drone_id, position=None, space_size=100):
        self.drone_id = drone_id
        self.space_size = space_size

        # Initial position
        if position is None:
            self.position = np.random.uniform(
                0,
                space_size,
                size=3
            ).astype(np.float32)
        else:
            self.position = np.array(
                position,
                dtype=np.float32
            )

        # Initial velocity
        self.velocity = np.zeros(
            3,
            dtype=np.float32
        )

    def move(self, action):

        # Store velocity
        self.velocity = np.asarray(
            action,
            dtype=np.float32
        )

        # Update position
        self.position += self.velocity

        # Keeping the drone inside the environment
        self.position = np.clip(
            self.position,
            0,
            self.space_size
        )

    def get_position(self):
        return self.position.copy()