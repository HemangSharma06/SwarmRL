from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from backend.Environment.SwarmEnv import SwarmEnv
from backend.training.centralized_env import CentralizedStatePettingZooEnv


ENV_NAME = "swarm_rl_env"


def env_creator(env_config):
    env = SwarmEnv(
        num_agents=env_config.get("num_agents", 2),
        space_size=env_config.get("space_size", 100),
        collision_threshold=env_config.get(
            "collision_threshold", 2.0
        ),
        max_episode_steps=env_config.get(
            "max_episode_steps", 500
        ),
        grid_size=env_config.get("grid_size", 20),
        obstacle_count=env_config.get("obstacle_count", 0),
        dynamic_obstacles=env_config.get("dynamic_obstacles", False),
        wind_strength=env_config.get("wind_strength", 0.0),
        wind_enabled=env_config.get("wind_enabled", True),
        curriculum_enabled=env_config.get("curriculum_enabled", False),
    )

    return CentralizedStatePettingZooEnv(env)

def register_swarm_env():
    register_env(ENV_NAME, env_creator)