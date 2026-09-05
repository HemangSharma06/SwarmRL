import os
import numpy as np
import torch
import ray
from backend.Environment.SwarmEnv import SwarmEnv
from backend.training.rllib import register_swarm_env, env_creator
from backend.training.mappo_config import get_mappo_config

def evaluate(
    checkpoint_path=None,
    num_agents=2,
    space_size=100,
    collision_threshold=2.0,
    max_episode_steps=500,
    num_episodes=5,
    render=False
):
    
    if checkpoint_path is None:
        checkpoint_path = "./checkpoints"
        # Find latest checkpoint
        if os.path.isdir(checkpoint_path):
            checkpoints = [
                d for d in os.listdir(checkpoint_path)
                if d.startswith("checkpoint_")
            ]
            if checkpoints:
                checkpoint_path = os.path.join(
                    checkpoint_path,
                    sorted(checkpoints)[-1]
                )
            else:
                print("No checkpoints found!")
                return None

    checkpoint_path = os.path.abspath(checkpoint_path)
    
    print(f"\nEvaluating checkpoint: {checkpoint_path}")
    print(f"  Agents: {num_agents}")
    print(f"  Episodes: {num_episodes}\n")
    
    ray.init(ignore_reinit_error=True)
    
    config = get_mappo_config(
        num_agents=num_agents,
        space_size=space_size,
        collision_threshold=collision_threshold,
        max_episode_steps=max_episode_steps
    )
    
    algorithm = config.build_algo()
    algorithm.restore(checkpoint_path)
    policy_module = algorithm.get_module("shared_policy")
    
    # Create environment
    env = SwarmEnv(
        num_agents=num_agents,
        space_size=space_size,
        collision_threshold=collision_threshold,
        max_episode_steps=max_episode_steps
    )
    
    # Initialize metrics
    episode_rewards = []
    episode_lengths = []
    total_collisions = 0
    
    print("Running evaluation episodes...\n")
    
    for episode in range(num_episodes):
        observations, infos = env.reset(seed=42 + episode)
        
        episode_reward = {agent: 0.0 for agent in env.agents}
        episode_length = 0
        episode_collisions = 0
        
        done = False
        
        while not done:
            # Get actions from trained policy
            actions = {}
            
            for agent in env.agents:
                observation = torch.as_tensor(
                    observations[agent], dtype=torch.float32
                ).unsqueeze(0)
                output = policy_module.forward_inference({"obs": observation})
                actions[agent] = (
                    output["actions"][0].detach().cpu().numpy()
                )
            
            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(
                actions
            )
            
            # Update metrics
            for agent in env.agents:
                episode_reward[agent] += rewards[agent]
                
                if infos[agent]["collision"]:
                    episode_collisions += 1
            
            episode_length += 1
            
            # Check if episode is done
            all_done = all(terminations.values()) or all(
                truncations.values()
            )
            if all_done:
                done = True
        
        total_reward = sum(episode_reward.values())
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        total_collisions += episode_collisions
        
        if render:
            print(
                f"Episode {episode + 1} | "
                f"Reward: {total_reward:.2f} | "
                f"Length: {episode_length} | "
                f"Collisions: {episode_collisions}"
            )
    
    # Compute statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    avg_collisions = total_collisions / num_episodes
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Average Episode Reward: {avg_reward:.2f} +/- {std_reward:.2f}")
    print(f"Average Episode Length: {avg_length:.2f}")
    print(f"Average Collisions per Episode: {avg_collisions:.2f}")
    print(f"Total Collisions: {total_collisions}")
    print("="*50 + "\n")
    
    algorithm.stop()
    ray.shutdown()
    
    metrics = {
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "avg_length": avg_length,
        "avg_collisions": avg_collisions,
        "total_collisions": total_collisions
    }
    
    return metrics

if __name__ == "__main__":
    evaluate(
        num_agents=2,
        num_episodes=5,
        render=True
    )
