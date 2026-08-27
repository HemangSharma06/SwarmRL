from ray.rllib.algorithms.ppo import PPOConfig

from backend.training.rllib import ENV_NAME
from backend.training.rllib import register_swarm_env

def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    return "shared_policy"

def get_mappo_config(
    num_agents=2,
    space_size=100,
    collision_threshold=2.0
):
    register_swarm_env()

    config = (
        PPOConfig()
        .environment(
            env=ENV_NAME,
            env_config={
                "num_agents": num_agents,
                "space_size": space_size,
                "collision_threshold": collision_threshold
            }
        )
        .framework("torch")
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=policy_mapping_fn
        )
        .training(
            gamma=0.99,
            lambda_=0.95,
            lr=3e-4,
            train_batch_size_per_learner=4000,
            minibatch_size=128,
            num_epochs=10,
            clip_param=0.2
        )
        .env_runners(
            num_env_runners=0
        )
    )

    return config