import torch.nn as nn

class ActorNetwork(nn.Module):
    def __init__(
        self,
        observation_dim=7,
        action_dim=3,
        hidden_dim=128
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, observation):
        """
        Generate continuous actions from
        the agent's local observation.

        Output range:
            [-1, 1]
        """
        return self.network(observation)


class CentralizedCritic(nn.Module):
    def __init__(
        self,
        global_observation_dim,
        hidden_dim=128
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                global_observation_dim,
                hidden_dim
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, global_observation):
        """
        Estimate the value of the global state.
        """

        return self.network(global_observation)