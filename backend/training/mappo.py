import torch
import torch.nn as nn

from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.columns import Columns

from backend.agents.network import (
    ActorNetwork,
    CentralizedCritic
)


class MAPPO_RLModule(TorchRLModule, ValueFunctionAPI):

    def setup(self):

        local_space = self.observation_space
        if hasattr(self.observation_space, "spaces"):
            local_space = self.observation_space.spaces["local_obs"]
        self.observation_dim = local_space.shape[0]

        self.action_dim = (
            self.action_space.shape[0]
        )

        self.actor = ActorNetwork(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim
        )

        model_config = self.model_config or {}
        if isinstance(model_config, dict):
            configured_global_dim = model_config.get("global_state_dim", self.observation_dim)
        else:
            configured_global_dim = getattr(model_config, "global_state_dim", self.observation_dim)
        self.global_state_dim = int(configured_global_dim)
        self.critic = CentralizedCritic(
            global_observation_dim=self.global_state_dim
        )
        
        # For continuous actions, we need to output mean and log_std
        # Actor network outputs [-1, 1] bounded actions via Tanh
        # We'll use a fixed log_std for simplicity
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

    def compute_values(self, batch, embeddings=None):
        return self.critic(self._critic_input(batch)).squeeze(-1)

    def _local_input(self, batch):
        observations = batch["obs"]
        if isinstance(observations, dict):
            observations = observations["local_obs"]
        return observations.float()

    def _critic_input(self, batch):
        observations = batch["obs"]
        global_state = observations.get("global_state") if isinstance(observations, dict) else batch.get("global_state")
        if global_state is None:
            infos = batch.get("infos")
            if isinstance(infos, dict):
                global_state = infos.get("global_state")
        if global_state is None:
            return self._local_input(batch)
        return torch.as_tensor(global_state, dtype=torch.float32, device=self.log_std.device)

    def _forward_inference(self, batch, **kwargs):

        observations = self._local_input(batch)

        actions = self.actor(observations)
        return {
            Columns.ACTIONS: actions
        }

    def _forward_exploration(self, batch, **kwargs):

        observations = self._local_input(batch)
        
        # For exploration, return distribution parameters
        # Actor outputs mean actions (tanh-bounded to [-1,1])
        mu = self.actor(observations)
        
        # Expand log_std to batch size
        batch_size = mu.shape[0]
        log_std = self.log_std.unsqueeze(0).expand(batch_size, -1)
        
        # Concatenate mean and log_std as action_dist_inputs
        # This will be used by RLlib to create Normal distribution
        action_dist_inputs = torch.cat([mu, log_std], dim=-1)
        action_dist = self.get_exploration_action_dist_cls()(
            mu,
            torch.exp(log_std)
        )
        sampled_actions, _ = action_dist.sample_and_logp()
        actions = torch.tanh(sampled_actions)
        action_logp = action_dist.logp(actions)
        values = self.compute_values(batch)

        return {
            Columns.ACTIONS: actions,
            Columns.ACTION_LOGP: action_logp,
            Columns.ACTION_DIST_INPUTS: action_dist_inputs,
            Columns.VF_PREDS: values.squeeze(-1)
        }

    def _forward_train(self, batch, **kwargs):

        observations = self._local_input(batch)
        
        # For training, return distribution parameters
        mu = self.actor(observations)
        
        batch_size = mu.shape[0]
        log_std = self.log_std.unsqueeze(0).expand(batch_size, -1)
        
        action_dist_inputs = torch.cat([mu, log_std], dim=-1)

        values = self.compute_values(batch)

        return {
            Columns.ACTIONS: mu,
            Columns.ACTION_DIST_INPUTS: action_dist_inputs,
            Columns.VF_PREDS: values.squeeze(-1)
        }