"""
dqn_agent.py

Contains a class that describes the DQN agent.

The following DQNAgent class was based on the following reference.
Reference: https://github.com/saashanair/rl-series/tree/master/dqn 
"""

import random
import numpy as np
import torch
import torch.nn.functional as F

from dqn.model import DQNNet
from dqn.replay_memory import ReplayMemory


class DQNAgent:
    """
    Defines the DQN agent, including training methods.
    """

    def __init__(
        self,
        device,
        state_size,
        action_size,
        discount=0.99,
        eps_max=1.0,
        eps_min=0.01,
        eps_decay=0.995,
        memory_capacity=5000,
        lr=1e-3,
        train_mode=True,
    ):

        self.device = device

        # defines the epsilon-greedy exploration strategy
        self.epsilon = eps_max
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay

        # defines how far-sighted the agent should be
        self.discount = discount

        # defines the size of the state vectors and number of possible actions
        self.state_size = state_size
        self.action_size = action_size

        # defines instances of the network for current policy and its target
        self.policy_net = DQNNet(self.state_size, self.action_size, lr).to(self.device)
        self.target_net = DQNNet(self.state_size, self.action_size, lr).to(self.device)
        self.target_net.eval()  # since no learning is performed on the target net
        if not train_mode:
            self.policy_net.eval()

        # defines instance of the replay buffer
        self.memory = ReplayMemory(capacity=memory_capacity)

    def update_target_net(self):
        """
        Copy the weights from the current policy net into the target net.
        """

        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        """
        Updates the epsilon value (by slowly reducing it based on the decay).
        """

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def select_action(self, state):
        """
        Selects an action using the epsilon-greedy exploration strategy for a given state.

        During training, returns a randomly sampled action or a greedy action (predicted by the policy network), based on the epsilon value.
        During testing, returns action predicted by the policy network
        """

        # perform random action if the random value is less than epsilon
        if (
            random.random() <= self.epsilon
        ):  # amount of exploration reduces with the epsilon value
            return random.randrange(self.action_size)

        if not torch.is_tensor(state):
            state = torch.tensor([state], dtype=torch.float32).to(self.device)

        # pick the action with maximum Q-value as per the policy Q-network
        with torch.no_grad():
            action = self.policy_net.forward(state)
        return torch.argmax(
            action
        ).item()  # since actions are discrete, return index that has highest Q

    def learn(self, batchsize):
        """
        Performs updates on the neural network.
        """

        # randomly select batchsize samples from the experience replay memory
        if len(self.memory) < batchsize:
            return
        states, actions, next_states, rewards, dones = self.memory.sample(
            batchsize, self.device
        )

        # get q values of the actions that were taken, i.e calculate qpred;
        # actions vector has to be explicitly reshaped to nx1-vector
        q_pred = self.policy_net.forward(states).gather(1, actions.view(-1, 1))

        # calculate target q-values, such that yj = rj + q(s', a'), but if current state is a terminal state, then yj = rj
        q_target = (
            self.target_net.forward(next_states).max(dim=1).values
        )  # because max returns data structure with values and indices
        q_target[
            dones
        ] = 0.0  # setting Q(s',a') to 0 when the current state is a terminal state
        y_j = rewards + (self.discount * q_target)
        y_j = y_j.view(-1, 1)

        # calculate the loss as the mean-squared error of yj and qpred
        self.policy_net.optimizer.zero_grad()
        loss = F.mse_loss(y_j, q_pred).mean()
        loss.backward()
        self.policy_net.optimizer.step()

    def save_model(self, filename):
        """
        Saves the policy network model.
        """

        self.policy_net.save_model(filename)

    def load_model(self, filename):
        """
        Loads the policy network model.
        """

        self.policy_net.load_model(filename=filename, device=self.device)
