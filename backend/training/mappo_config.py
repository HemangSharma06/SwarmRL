from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from backend.training.rllib import (
    ENV_NAME,
    register_swarm_env
)
from backend.training.mappo import MAPPO_RLModule


def policy_mapping_fn(
    agent_id,
    episode=None,
    worker=None,
    **kwargs
):
    return "shared_policy"


def get_mappo_config(
    num_agents=2,
    space_size=100,
    collision_threshold=2.0,
    max_episode_steps=500,
    grid_size=20,
    obstacle_count=0,
    dynamic_obstacles=False,
    wind_strength=0.0,
    wind_enabled=True,
    curriculum_enabled=False,
    train_batch_size=4000,
    minibatch_size=128
):
    register_swarm_env()

    config = (
        PPOConfig()
        .environment(
            env=ENV_NAME,
            env_config={
                "num_agents": num_agents,
                "space_size": space_size,
                "collision_threshold": collision_threshold,
                "max_episode_steps": max_episode_steps,
                "grid_size": grid_size,
                "obstacle_count": obstacle_count,
                "dynamic_obstacles": dynamic_obstacles,
                "wind_strength": wind_strength,
                "wind_enabled": wind_enabled,
                "curriculum_enabled": curriculum_enabled,
            }
        )
        .framework("torch")
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=policy_mapping_fn
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=MAPPO_RLModule,
                model_config={"global_state_dim": num_agents * 4}
            )
        )
        .training(
            gamma=0.99,
            lambda_=0.95,
            lr=3e-4,
            train_batch_size_per_learner=train_batch_size,
            minibatch_size=min(minibatch_size, train_batch_size),
            num_epochs=10,
            clip_param=0.2
        )
        .env_runners(
            num_env_runners=0,
            create_local_env_runner=True
        )
    )

    return config