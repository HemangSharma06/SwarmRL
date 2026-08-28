import numpy as np

class GlobalState:
    def __init__(self, env):
        self.env = env

    def get_state(self):
        states = []

        for agent in self.env.agents:

            position = self.env.drones[
                agent
            ].get_position()

            _, nearest_distance = (
                self.env.get_nearest_drone(agent)
            )

            states.extend([
                position[0],
                position[1],
                position[2],
                nearest_distance
            ])
        return np.asarray(
            states,
            dtype=np.float32
        )

    def get_state_dim(self):
        return len(self.env.agents) * 4