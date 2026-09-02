import torch
import torch.nn as nn

from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.columns import Columns

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
        
        # For continuous actions, we need to output mean and log_std
        # Actor network outputs [-1, 1] bounded actions via Tanh
        # We'll use a fixed log_std for simplicity
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

    def _forward_inference(self, batch, **kwargs):

        observations = batch["obs"].float()

        actions = self.actor(observations)
        return {
            Columns.ACTIONS: actions
        }

    def _forward_exploration(self, batch, **kwargs):

        observations = batch["obs"].float()
        
        # For exploration, return distribution parameters
        # Actor outputs mean actions (tanh-bounded to [-1,1])
        mu = self.actor(observations)
        
        # Expand log_std to batch size
        batch_size = mu.shape[0]
        log_std = self.log_std.unsqueeze(0).expand(batch_size, -1)
        
        # Concatenate mean and log_std as action_dist_inputs
        # This will be used by RLlib to create Normal distribution
        action_dist_inputs = torch.cat([mu, log_std], dim=-1)

        return {
            Columns.ACTIONS: mu,
            Columns.ACTION_DIST_INPUTS: action_dist_inputs
        }

    def _forward_train(self, batch, **kwargs):

        observations = batch["obs"].float()
        
        # For training, return distribution parameters
        mu = self.actor(observations)
        
        batch_size = mu.shape[0]
        log_std = self.log_std.unsqueeze(0).expand(batch_size, -1)
        
        action_dist_inputs = torch.cat([mu, log_std], dim=-1)

        values = self.critic(observations)

        return {
            Columns.ACTIONS: mu,
            Columns.ACTION_DIST_INPUTS: action_dist_inputs,
            Columns.VF_PREDS: values.squeeze(-1)
        }