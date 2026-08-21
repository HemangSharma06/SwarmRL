import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from backend.Environment.drone import Drone
from backend.Environment.reward import SwarmReward


class SwarmEnv(ParallelEnv):

    metadata = {
        "name": "swarm_env_v0"
    }

    def __init__(
        self,
        num_agents=2,
        space_size=100,
        collision_threshold=2.0
    ):
        super().__init__()

        self.space_size = space_size
        self.collision_threshold = collision_threshold

        # Reward system
        self.reward_system = SwarmReward()

        # Agent names
        self.possible_agents = [
            f"drone_{i}"
            for i in range(num_agents)
        ]

        self.agents = self.possible_agents.copy()

        # Action:
        # [velocity_x, velocity_y, velocity_z]
        self.action_spaces = {
            agent: spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }

        # Observation:
        #
        # [x, y, z,
        #  nearest_dx, nearest_dy, nearest_dz,
        #  nearest_distance]
        #
        self.observation_spaces = {
            agent: spaces.Box(
                low=np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        -space_size,
                        -space_size,
                        -space_size,
                        0.0
                    ],
                    dtype=np.float32
                ),
                high=np.array(
                    [
                        space_size,
                        space_size,
                        space_size,
                        space_size,
                        space_size,
                        space_size,
                        np.sqrt(3 * space_size ** 2)
                    ],
                    dtype=np.float32
                ),
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

        # Create drones
        self.drones = {
            agent: Drone(
                drone_id=agent,
                space_size=self.space_size
            )
            for agent in self.agents
        }

        # Initial observations
        observations = {
            agent: self.get_observation(agent)
            for agent in self.agents
        }

        infos = {
            agent: {}
            for agent in self.agents
        }

        return observations, infos

    def calculate_distance(self, agent_a, agent_b):

        position_a = self.drones[agent_a].get_position()
        position_b = self.drones[agent_b].get_position()

        distance = np.linalg.norm(
            position_a - position_b
        )

        return float(distance)

    def get_nearest_drone(self, agent):

        nearest_drone = None
        nearest_distance = float("inf")

        for other_agent in self.agents:

            if other_agent == agent:
                continue

            distance = self.calculate_distance(
                agent,
                other_agent
            )

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_drone = other_agent

        return nearest_drone, nearest_distance

    def get_observation(self, agent):

        position = self.drones[agent].get_position()

        nearest_drone, nearest_distance = (
            self.get_nearest_drone(agent)
        )

        # If there are no other drones
        if nearest_drone is None:

            relative_position = np.zeros(
                3,
                dtype=np.float32
            )

            nearest_distance = 0.0

        else:

            nearest_position = (
                self.drones[nearest_drone].get_position()
            )

            relative_position = (
                nearest_position - position
            )

        observation = np.concatenate(
            [
                position,
                relative_position,
                np.array(
                    [nearest_distance],
                    dtype=np.float32
                )
            ]
        )

        return observation.astype(np.float32)

    def check_collision(self, agent_a, agent_b):

        distance = self.calculate_distance(
            agent_a,
            agent_b
        )

        return distance <= self.collision_threshold

    def step(self, actions):

        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # Move drones
        for agent, action in actions.items():

            action = np.asarray(
                action,
                dtype=np.float32
            )

            self.drones[agent].move(action)

        # Detect collisions
        collisions = set()

        for i, agent_a in enumerate(self.agents):

            for agent_b in self.agents[i + 1:]:

                if self.check_collision(
                    agent_a,
                    agent_b
                ):
                    collisions.add(agent_a)
                    collisions.add(agent_b)

        # Calculate rewards
        for agent in self.agents:

            nearest_drone, nearest_distance = (
                self.get_nearest_drone(agent)
            )

            collision = agent in collisions

            rewards[agent] = (
                self.reward_system.calculate_reward(
                    collision=collision,
                    distance_to_nearest=nearest_distance
                )
            )

            terminations[agent] = False
            truncations[agent] = False

            infos[agent] = {
                "nearest_drone": nearest_drone,
                "nearest_distance": nearest_distance,
                "collision": collision
            }

        # New observations
        observations = {
            agent: self.get_observation(agent)
            for agent in self.agents
        }

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos
        )