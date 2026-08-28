import torch

from ray.rllib.core.rl_module.torch import TorchRLModule

from backend.agents.network import (
    ActorNetwork,
    CentralizedCritic
)


class MAPPO_RLModule(TorchRLModule):

    def setup(self):

        self.observation_dim = (
            self.observation_space.shape[0]
        )

        self.action_dim = (
            self.action_space.shape[0]
        )

        self.actor = ActorNetwork(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim
        )

        self.critic = CentralizedCritic(
            global_observation_dim=self.observation_dim
        )

    def _forward_inference(self, batch):

        observations = batch["obs"].float()

        actions = self.actor(observations)

        return {
            "actions": actions
        }

    def _forward_exploration(self, batch):

        observations = batch["obs"].float()

        actions = self.actor(observations)

        return {
            "actions": actions
        }

    def _forward_train(self, batch):

        observations = batch["obs"].float()

        actions = self.actor(observations)
        values = self.critic(observations)

        return {
            "actions": actions,
            "vf_preds": values.squeeze(-1)
        }