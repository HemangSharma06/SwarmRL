from __future__ import annotations

import numpy as np
from gymnasium import spaces
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv


class CentralizedStatePettingZooEnv(ParallelPettingZooEnv):
    """RLlib adapter carrying local observations and a shared global state."""

    def __init__(self, env):
        super().__init__(env)
        global_dim = env.get_global_state_dim()
        global_high = np.tile(
            [env.space_size, env.space_size, env.space_size, np.sqrt(3 * env.space_size**2)],
            len(env.agents),
        ).astype(np.float32)
        global_space = spaces.Box(
            low=np.zeros(global_dim, dtype=np.float32),
            high=global_high,
            dtype=np.float32,
        )
        self.observation_spaces = {
            agent: spaces.Dict({
                "local_obs": env.observation_space(agent),
                "global_state": global_space,
            })
            for agent in env.possible_agents
        }
        self.observation_space = spaces.Dict(self.observation_spaces)

    @staticmethod
    def _wrap_observations(observations, infos):
        global_state = next(iter(infos.values()))["global_state"]
        return {
            agent: {
                "local_obs": observation,
                "global_state": global_state.copy(),
            }
            for agent, observation in observations.items()
        }

    def reset(self, *, seed=None, options=None):
        observations, infos = self.par_env.reset(seed=seed, options=options)
        return self._wrap_observations(observations, infos), infos

    def step(self, action_dict):
        observations, rewards, terminateds, truncateds, infos = self.par_env.step(action_dict)
        observations = self._wrap_observations(observations, infos)
        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = all(truncateds.values())
        return observations, rewards, terminateds, truncateds, infos
