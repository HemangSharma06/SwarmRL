import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from backend.Environment.drone import Drone


class SwarmEnv(ParallelEnv):

    metadata = {
        "name": "swarm_env_v0"
    }

    def __init__(self, num_agents=2, space_size=100):
        super().__init__()
        self.space_size = space_size

        # Agent names
        self.possible_agents = [
            f"drone_{i}"
            for i in range(num_agents)
        ]

        self.agents = self.possible_agents.copy()

        # Action space: [velocity_x, velocity_y, velocity_z]
        self.action_spaces = {
            agent: spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }

        # Observation space: [x, y, z]
        self.observation_spaces = {
            agent: spaces.Box(
                low=0.0,
                high=space_size,
                shape=(3,),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }
        # Drone objects
        self.drones = {}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Reset active agents
        self.agents = self.possible_agents.copy()

        # Creating drones
        self.drones = {
            agent: Drone(
                drone_id=agent,
                space_size=self.space_size
            )
            for agent in self.agents
        }

        # Initial observations
        observations = {
            agent: self.drones[agent].get_position()
            for agent in self.agents
        }
        infos = {
            agent: {}
            for agent in self.agents
        }
        return observations, infos

    def step(self, actions):

        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # Moving the each drone
        for agent, action in actions.items():
            action = np.asarray(
                action,
                dtype=np.float32
            )
            self.drones[agent].move(action)

        for agent in self.agents:
            rewards[agent] = 0.0
            terminations[agent] = False
            truncations[agent] = False
            infos[agent] = {}

        observations = {
            agent: self.drones[agent].get_position()
            for agent in self.agents
        }

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos
        )