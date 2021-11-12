import numpy as np


class Memory(object):
    def __init__(self, MAX_SIZE, INPUT_SHAPE, NUM_ACTIONS):
        self.MEMORY_SIZE = MAX_SIZE
        self.counter = 0
        self.state_memory = np.zeros((self.MEMORY_SIZE, *INPUT_SHAPE), dtype=np.float32)
        self.action_memory = np.zeros(self.MEMORY_SIZE, dtype=np.int64)
        self.reward_memory = np.zeros(self.MEMORY_SIZE, dtype=np.float32)

    def store_transition(self, env_state, action, reward):
        index = self.counter % self.MEMORY_SIZE
        self.state_memory[index] = env_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.counter += 1

    def sample_buffer(self, batch_size):
        MAX_MEMORY_SIZE = min(self.counter, self.MEMORY_SIZE)
        batch = np.random.choice(MAX_MEMORY_SIZE, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]

        return states, actions, rewards
