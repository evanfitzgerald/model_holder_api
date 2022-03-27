"""
replay_memory.py

Contains a class that describes the experience replay buffer.

The following ReplayMemory class was based on the following reference.
Reference: https://github.com/saashanair/rl-series/tree/master/dqn 
"""

import random
import numpy as np
import torch


class ReplayMemory:
    """
    Defines the replay buffer, which stoes the agent's experinces for off-policy learning
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer_state = []
        self.buffer_action = []
        self.buffer_next_state = []
        self.buffer_reward = []
        self.buffer_done = []
        self.idx = 0

    def store(self, state, action, next_state, reward, done):
        """
        Adds an experience to the memory.

        An experience is made up of the following 5-tuple:
            (state, action, next_state, reward, done)
        """

        if len(self.buffer_state) < self.capacity:
            self.buffer_state.append(state)
            self.buffer_action.append(action)
            self.buffer_next_state.append(next_state)
            self.buffer_reward.append(reward)
            self.buffer_done.append(done)
        else:
            self.buffer_state[self.idx] = state
            self.buffer_action[self.idx] = action
            self.buffer_next_state[self.idx] = next_state
            self.buffer_reward[self.idx] = reward
            self.buffer_done[self.idx] = done

        # for circular memory
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size, device):
        """
        Randomly selects batch_size samples from the memory and returns them as tensors.
        """

        indices_to_sample = random.sample(range(len(self.buffer_state)), batch_size)

        states = (
            torch.from_numpy(np.array(self.buffer_state)[indices_to_sample])
            .float()
            .to(device)
        )
        actions = torch.from_numpy(np.array(self.buffer_action)[indices_to_sample]).to(
            device
        )
        next_states = (
            torch.from_numpy(np.array(self.buffer_next_state)[indices_to_sample])
            .float()
            .to(device)
        )
        rewards = (
            torch.from_numpy(np.array(self.buffer_reward)[indices_to_sample])
            .float()
            .to(device)
        )
        dones = torch.from_numpy(np.array(self.buffer_done)[indices_to_sample]).to(
            device
        )

        return states, actions, next_states, rewards, dones

    def __len__(self):
        """
        Returns the number of elements in the replay memory.
        """

        return len(self.buffer_state)
