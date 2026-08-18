import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv


class SwarmEnv(ParallelEnv):
    metadata = {"name": "swarm_env_v0"}
    
    def __init__(self, num_agents=2, space_size=100):
        super().__init__()

        self.num_agents = num_agents
        self.space_size = space_size

        # Agent names
        self.possible_agents = [
            f"drone_{i}" for i in range(num_agents)
        ]

        self.agents = self.possible_agents.copy()

        self.action_spaces = {
            agent: spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }

        self.observation_spaces = {
            agent: spaces.Box(
                low=0.0,
                high=space_size,
                shape=(3,),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }

        self.positions = {}

    def reset(self, seed=None, options=None):

        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents.copy()

        self.positions = {
            agent: np.random.uniform(
                0,
                self.space_size,
                size=3
            ).astype(np.float32)
            for agent in self.agents
        }

        observations = {
            agent: self.positions[agent].copy()
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

        for agent, action in actions.items():
            action = np.asarray(action, dtype=np.float32)

            self.positions[agent] += action
            self.positions[agent] = np.clip(
                self.positions[agent],
                0,
                self.space_size
            )
        for agent in self.agents:

            rewards[agent] = 0.0

            terminations[agent] = False
            truncations[agent] = False

            infos[agent] = {}

        observations = {
            agent: self.positions[agent].copy()
            for agent in self.agents
        }

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos
        )