import os
import ray
from backend.training.mappo_config import get_mappo_config

def train(
    num_agents=2,
    space_size=100,
    collision_threshold=2.0,
    max_episode_steps=500,
    iterations=10,
    checkpoint_freq=5,
    checkpoint_dir="./checkpoints",
    train_batch_size=4000
):
    """
    Train MAPPO algorithm on SwarmEnv.
    
    Args:
        num_agents: Number of drone agents
        space_size: Size of environment
        collision_threshold: Collision detection distance
        max_episode_steps: Maximum steps per episode
        iterations: Number of training iterations
        checkpoint_freq: Save checkpoint every N iterations
        checkpoint_dir: Directory to save checkpoints
    """
    
    ray.init(ignore_reinit_error=True)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    config = get_mappo_config(
        num_agents=num_agents,
        space_size=space_size,
        collision_threshold=collision_threshold,
        max_episode_steps=max_episode_steps,
        train_batch_size=train_batch_size
    )
    
    algorithm = config.build_algo()
    
    print(f"\nStarting MAPPO training...")
    print(f"  Agents: {num_agents}")
    print(f"  Space size: {space_size}")
    print(f"  Max episode steps: {max_episode_steps}")
    print(f"  Training iterations: {iterations}\n")
    
    for iteration in range(iterations):
        try:
            result = algorithm.train()
            
            episode_return = result.get(
                "env_runners",
                {}
            ).get(
                "episode_return_mean",
                0.0
            )
            
            episode_len = result.get(
                "env_runners",
                {}
            ).get(
                "episode_len_mean",
                0.0
            )
            
            policy_loss = result.get(
                "info",
                {}
            ).get(
                "learner",
                {}
            ).get(
                "default_policy",
                {}
            ).get(
                "policy_loss",
                0.0
            )
            
            print(
                f"Iteration {iteration + 1}/{iterations} | "
                f"Episode Return: {episode_return:.2f} | "
                f"Episode Length: {episode_len:.2f}"
            )
            
            # Save checkpoint
            if (iteration + 1) % checkpoint_freq == 0:
                checkpoint_path = algorithm.save(checkpoint_dir)
                print(f"  Checkpoint saved to {checkpoint_path}")
                
        except Exception as e:
            print(f"Error during training iteration {iteration + 1}: {e}")
            raise
    
    # Final checkpoint
    final_checkpoint = algorithm.save(checkpoint_dir)
    print(f"\nFinal checkpoint saved to {final_checkpoint}")
    
    algorithm.stop()
    ray.shutdown()
    
    print("Training completed successfully!")
    return final_checkpoint

if __name__ == "__main__":
    train(
        num_agents=2,
        iterations=10
    )