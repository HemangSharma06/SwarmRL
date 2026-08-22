class Agent:

    def __init__(self, agent_id, observation_space, action_space):
        """
        Parameters:
            agent_id: Unique identifier of the agent.
            observation_space: Gymnasium observation space.
            action_space: Gymnasium action space.
        """

        self.agent_id = agent_id
        self.observation_space = observation_space
        self.action_space = action_space

        # Current state of the agent
        self.observation = None
        self.action = None

        # Learning information
        self.reward = 0.0
        self.done = False

    def select_action(self, observation):
        """
        Select an action based on the current observation.
        This method will be implemented by specific
        reinforcement learning agents.
        """
        raise NotImplementedError(
            "select_action() must be implemented by the agent."
        )

    def observe(self, observation):
        """
        Store the current observation.
        """
        self.observation = observation

    def update(self, reward, done=False):
        """
        Update the agent using the received reward.
        """
        self.reward = reward
        self.done = done

    def reset(self):
        """
        Reset the agent's internal state.
        """
        self.observation = None
        self.action = None
        self.reward = 0.0
        self.done = False