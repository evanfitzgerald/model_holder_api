"""
model.py

Contains a class that describes the neural network model.

The following DQNNet class was based on the following reference.
Reference: https://github.com/saashanair/rl-series/tree/master/dqn 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQNNet(nn.Module):
    """
    Defines the neural network architecture for the DQN agent
    """

    def __init__(self, input_size, output_size, lr=1e-3):
        super(DQNNet, self).__init__()
        self.dense1 = nn.Linear(input_size, 400)
        self.dense2 = nn.Linear(400, 300)
        self.dense3 = nn.Linear(300, output_size)

        # Uses the Adam optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x

    def save_model(self, filename):
        """
        Saves the model parameters.
        """

        torch.save(self.state_dict(), filename)

    def load_model(self, filename, device):
        """
        Loads the model parameters.
        """

        self.load_state_dict(torch.load(filename, map_location=device))
