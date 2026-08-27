import ray
from backend.training.mappo_config import get_mappo_config

def train(
    num_agents=2,
    space_size=100,
    collision_threshold=2.0,
    iterations=10
):
    ray.init(ignore_reinit_error=True)
    config = get_mappo_config(
        num_agents=num_agents,
        space_size=space_size,
        collision_threshold=collision_threshold
    )
    algorithm = config.build_algo()
    
    for iteration in range(iterations):
        result = algorithm.train()
        reward = result.get(
            "env_runners",
            {}
        ).get(
            "episode_return_mean",
            None
        )
        print(
            f"Iteration {iteration + 1}: "
            f"reward={reward}"
        )
    algorithm.stop()
    ray.shutdown()

if __name__ == "__main__":
    train(
        num_agents=2,
        iterations=10
    )