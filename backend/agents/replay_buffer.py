import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=100000):
        """
        Replay buffer for storing reinforcement learning experiences.

        Each experience contains:

            state
            action
            reward
            next_state
            done
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        """
        Store one experience in the replay buffer.
        """
        experience = (
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done)
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.position = (
            (self.position + 1)
            % self.capacity
        )

    def sample(self, batch_size):
        """
        Sample a random batch of experiences.
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                "Batch size cannot be greater than "
                "the number of stored experiences."
            )
        batch = random.sample(
            self.buffer,
            batch_size
        )

        states, actions, rewards, next_states, dones = zip(
            *batch
        )
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32)
        )

    def __len__(self):
        """
        Return the current number of stored experiences.
        """
        return len(self.buffer)

    def clear(self):
        """
        Clear all stored experiences.
        """
        self.buffer.clear()
        self.position = 0