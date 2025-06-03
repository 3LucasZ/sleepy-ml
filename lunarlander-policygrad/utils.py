import os

import torch

dir_path = os.path.dirname(os.path.realpath(__file__))
model_filepath = os.path.join(dir_path, "model.pth")


def best_model_filepath(num):
    return os.path.join(dir_path, f"best_model_{num}.pth")


class StochasticPolicy(torch.nn.Module):
    def __init__(self, n_states, n_actions):
        super(StochasticPolicy, self).__init__()
        self.fc1 = torch.nn.Linear(n_states, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return torch.softmax(x, dim=1)

    def selectAction(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        actionProbs = self.forward(state)
        m = torch.distributions.Categorical(actionProbs)
        action = m.sample()
        return action.item(), m.log_prob(action)
