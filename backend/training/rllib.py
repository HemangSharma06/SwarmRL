from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from backend.Environment.SwarmEnv import SwarmEnv


ENV_NAME = "swarm_rl_env"


def env_creator(env_config):
    env = SwarmEnv(
        num_agents=env_config.get("num_agents", 2),
        space_size=env_config.get("space_size", 100),
        collision_threshold=env_config.get(
            "collision_threshold", 2.0
        )
    )

    return ParallelPettingZooEnv(env)


def register_swarm_env():
    register_env(ENV_NAME, env_creator)