import numpy as np
import torch

from backend.agents.network import (
    ActorNetwork,
    CentralizedCritic
)

class MAPPOAgent:

    def __init__(
        self,
        agent_id,
        observation_dim=7,
        action_dim=3,
        global_observation_dim=None,
        hidden_dim=128,
        device=None
    ):
        """
        MAPPO agent containing a decentralized actor
        and a centralized critic.

        Parameters:
            agent_id:
                Unique drone identifier.

            observation_dim:
                Dimension of local observation.

            action_dim:
                Dimension of continuous action.

            global_observation_dim:
                Dimension of the global observation used
                by the centralized critic.

            hidden_dim:
                Number of neurons in hidden layers.

            device:
                PyTorch device.
        """
        self.agent_id = agent_id

        self.observation_dim = observation_dim
        self.action_dim = action_dim

        if global_observation_dim is None:
            global_observation_dim = observation_dim

        self.global_observation_dim = (
            global_observation_dim
        )

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        # Decentralized actor
        self.actor = ActorNetwork(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        # Centralized critic
        self.critic = CentralizedCritic(
            global_observation_dim=global_observation_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

    def select_action(self, observation):
        """
        Generate a continuous action using
        the decentralized actor.

        Parameters:
            observation:
                Local observation of the drone.

        Returns:
            Action as a NumPy array.
        """
        observation = np.asarray(
            observation,
            dtype=np.float32
        )
        observation_tensor = torch.tensor(
            observation,
            dtype=torch.float32,
            device=self.device
        )
        # Add batch dimension
        if observation_tensor.dim() == 1:
            observation_tensor = (
                observation_tensor.unsqueeze(0)
            )
        with torch.no_grad():
            action = self.actor(
                observation_tensor
            )

        action = action.squeeze(0)
        return action.cpu().numpy()

    def get_value(self, global_observation):
        """
        Estimate the value of the global state
        using the centralized critic.

        Parameters:
            global_observation:
                Global observation of the swarm.

        Returns:
            Estimated state value.
        """

        global_observation = np.asarray(
            global_observation,
            dtype=np.float32
        )
        observation_tensor = torch.tensor(
            global_observation,
            dtype=torch.float32,
            device=self.device
        )

        if observation_tensor.dim() == 1:
            observation_tensor = (
                observation_tensor.unsqueeze(0)
            )

        with torch.no_grad():

            value = self.critic(
                observation_tensor
            )

        return float(value.squeeze().cpu().item())

    def train_mode(self):
        """
        Put actor and critic into training mode.
        """
        self.actor.train()
        self.critic.train()

    def eval_mode(self):
        """
        Put actor and critic into evaluation mode.
        """
        self.actor.eval()
        self.critic.eval()

    def save(self, path):
        """
        Save actor and critic parameters.
        """
        torch.save(
            {
                "agent_id": self.agent_id,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict()
            },
            path
        )

    def load(self, path):
        """
        Load actor and critic parameters.
        """
        checkpoint = torch.load(
            path,
            map_location=self.device
        )
        self.actor.load_state_dict(
            checkpoint["actor"]
        )
        self.critic.load_state_dict(
            checkpoint["critic"]
        )