import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from backend.Environment.drone import Drone
from backend.Environment.reward import SwarmReward
from backend.Environment.Global import GlobalState
from backend.Environment.coverage import CoverageTracker
from backend.Environment.obstacle import DynamicObstacle
from backend.training.curriculum import CurriculumManager
from backend.simulation_state import SimulationState


class SwarmEnv(ParallelEnv):

    metadata = {
        "name": "swarm_env_v0"
    }

    def __init__(
        self,
        num_agents=2,
        space_size=100,
        collision_threshold=2.0,
        max_episode_steps=500,
        grid_size=20,
        obstacle_count=0,
        dynamic_obstacles=False,
        wind_strength=0.0,
        wind_enabled=True,
        curriculum_enabled=False
    ):
        super().__init__()

        self.space_size = space_size
        self.collision_threshold = collision_threshold
        self.max_episode_steps = max_episode_steps
        self.episode_step = 0
        self.episode = 0
        self.grid_size = grid_size
        self.obstacle_count = obstacle_count
        self.dynamic_obstacles = dynamic_obstacles
        self.wind_strength = wind_strength
        self.wind_enabled = wind_enabled
        self.wind = np.zeros(3, dtype=np.float32)
        self.obstacle_collision_count = 0
        self.collision_count = 0
        self.total_reward = 0.0
        self.last_rewards = {}
        self.last_collisions = set()
        self.last_obstacle_collisions = set()
        self.coverage_tracker = CoverageTracker(space_size, grid_size)
        self.curriculum = CurriculumManager(enabled=curriculum_enabled)
        self.obstacles = {}
        self.simulation_status = "idle"

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

        # Local observation:
        # [x, y, z,
        #  nearest_dx, nearest_dy, nearest_dz,
        #  nearest_distance]
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
                        np.sqrt(
                            3 * space_size ** 2
                        )
                    ],
                    dtype=np.float32
                ),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }

        # Drone objects
        self.drones = {}

        # Global state handler
        self.global_state = GlobalState(self)
        self.simulation_state = SimulationState(total_cells=self.coverage_tracker.total_cells)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):

        if seed is not None:
            np.random.seed(seed)

        level = self.curriculum.current_level
        obstacle_count = self.obstacle_count or level.obstacle_count
        dynamic_obstacles = self.dynamic_obstacles or level.dynamic_obstacles
        effective_wind_strength = self.wind_strength or level.wind_strength

        self.agents = self.possible_agents.copy()
        self.episode_step = 0
        if self.episode > 0:
            self.curriculum.advance()
        self.episode += 1
        self.collision_count = 0
        self.obstacle_collision_count = 0
        self.total_reward = 0.0
        self.coverage_tracker.reset()
        self.wind = np.zeros(3, dtype=np.float32)
        if self.wind_enabled and effective_wind_strength > 0:
            self.wind = np.random.uniform(-1.0, 1.0, size=3).astype(np.float32)
            self.wind = self.wind / max(np.linalg.norm(self.wind), 1e-6) * effective_wind_strength

        # Create drones
        self.drones = {
            agent: Drone(
                drone_id=agent,
                space_size=self.space_size
            )
            for agent in self.agents
        }

        self.obstacles = {
            f"obstacle_{index}": DynamicObstacle(
                obstacle_id=f"obstacle_{index}",
                position=np.random.uniform(0, self.space_size, size=3),
                radius=2.0,
                velocity=(np.random.uniform(-0.25, 0.25, size=3) if dynamic_obstacles else np.zeros(3)),
                space_size=self.space_size,
            )
            for index in range(obstacle_count)
        }
        self.simulation_status = "running"

        observations = {
            agent: self.get_observation(agent)
            for agent in self.agents
        }

        global_state = self.get_global_state()

        infos = {
            agent: {
                "global_state": global_state.copy()
            }
            for agent in self.agents
        }

        return observations, infos

    def calculate_distance(self, agent_a, agent_b):

        position_a = self.drones[
            agent_a
        ].get_position()

        position_b = self.drones[
            agent_b
        ].get_position()

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

        position = self.drones[
            agent
        ].get_position()

        nearest_drone, nearest_distance = (
            self.get_nearest_drone(agent)
        )

        if nearest_drone is None:

            relative_position = np.zeros(
                3,
                dtype=np.float32
            )

            nearest_distance = 0.0

        else:

            nearest_position = (
                self.drones[
                    nearest_drone
                ].get_position()
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

    def get_global_state(self):

        return self.global_state.get_state()

    def get_global_state_dim(self):

        return self.global_state.get_state_dim()

    def get_simulation_state(self):
        drones = []
        for agent, drone in self.drones.items():
            _, nearest_distance = self.get_nearest_drone(agent)
            drones.append({
                "id": agent,
                "position": drone.get_position().tolist(),
                "velocity": drone.velocity.tolist(),
                "reward": float(self.last_rewards.get(agent, 0.0)),
                "collision": agent in self.last_collisions or agent in self.last_obstacle_collisions,
                "nearest_distance": float(nearest_distance),
                "searched": self.coverage_tracker._cell_for_position(drone.get_position()) in self.coverage_tracker.visited_cells,
            })
        self.simulation_state = SimulationState(
            episode=self.episode,
            step=self.episode_step,
            simulation_status=self.simulation_status,
            coverage=self.coverage_tracker.coverage_percentage,
            explored_cells=self.coverage_tracker.explored_cells,
            total_cells=self.coverage_tracker.total_cells,
            collision_count=self.collision_count,
            obstacle_collision_count=self.obstacle_collision_count,
            reward=float(sum(self.last_rewards.values())),
            total_reward=self.total_reward,
            wind=self.wind.tolist(),
            drones=drones,
            obstacles=[obstacle.to_dict() for obstacle in self.obstacles.values()],
            curriculum_level=self.curriculum.current_level.level,
        )
        return self.simulation_state.to_dict()

    def check_collision(self, agent_a, agent_b):

        distance = self.calculate_distance(
            agent_a,
            agent_b
        )

        return distance <= self.collision_threshold

    def step(self, actions):

        self.episode_step += 1

        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for obstacle in self.obstacles.values():
            obstacle.move()

        # Move drones
        for agent, action in actions.items():

            action = np.asarray(
                action,
                dtype=np.float32
            )

            applied_wind = self.wind if self.wind_enabled else np.zeros(3, dtype=np.float32)
            self.drones[agent].move(action, wind=applied_wind)

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

        obstacle_collisions = {
            agent for agent, drone in self.drones.items()
            if any(obstacle.collides_with(drone.get_position()) for obstacle in self.obstacles.values())
        }
        self.collision_count += len(collisions) // 2
        self.obstacle_collision_count += len(obstacle_collisions)
        newly_explored_by_agent = {
            agent: self.coverage_tracker.visit(self.drones[agent].get_position())
            for agent in self.agents
        }
        newly_explored = sum(newly_explored_by_agent.values())
        self.last_collisions = collisions
        self.last_obstacle_collisions = obstacle_collisions

        # Global state after movement
        global_state = self.get_global_state()

        # Check if episode is truncated (max steps reached)
        episode_truncated = (
            self.episode_step >= self.max_episode_steps
        )

        # Calculate rewards
        for agent in self.agents:

            nearest_drone, nearest_distance = (
                self.get_nearest_drone(agent)
            )

            collision = agent in collisions or agent in obstacle_collisions

            rewards[agent] = (
                self.reward_system.calculate_reward(
                    collision=collision,
                    distance_to_nearest=nearest_distance,
                    newly_explored=newly_explored_by_agent[agent]
                )
            )
            self.total_reward += rewards[agent]
            self.last_rewards = rewards.copy()

            # Termination if collision occurs (permanent)
            terminations[agent] = collision

            # Truncation if episode max steps reached
            truncations[agent] = episode_truncated

            infos[agent] = {
                "nearest_drone": nearest_drone,
                "nearest_distance": nearest_distance,
                "collision": collision,
                "obstacle_collision": agent in obstacle_collisions,
                "newly_explored": int(newly_explored_by_agent[agent]),
                "global_state": global_state.copy()
            }
        observations = {
            agent: self.get_observation(agent)
            for agent in self.agents
        }

        if episode_truncated or any(terminations.values()):
            self.simulation_status = "completed"

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos
        )