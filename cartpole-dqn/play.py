from collections import deque, namedtuple
import random
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import torch
from utils import *


# Settings
episodes = 50
epsilon = 0.01
model = -1  # -1 means last trained model, 0-7 are previous well-trained models


# Neural network to find Q-values
# input: state
# output: Q-values for all possible actions
class DQN(torch.nn.Module):
    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(n_states, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


env = gym.make('CartPole-v1',
               render_mode="human")
policy_net = DQN(env.observation_space.shape[0],  env.action_space.n)
buffer = torch.load(
    best_model_filepath(model) if model > -1 else model_filepath, weights_only=True)
policy_net.load_state_dict(buffer)

outcomes = np.zeros((episodes))


def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = policy_net(state)
        return torch.argmax(q_values).item()


def play():
    for i in range(episodes):
        state, _ = env.reset()
        rewardTot = 0
        while True:
            # pick an action
            action = select_action(state, epsilon)

            # perform the action
            new_state, reward, done, _, _ = env.step(action)
            state = new_state
            rewardTot += reward

            if done:
                outcomes[i] = rewardTot
                break


def plot():
    # Plot outcomes
    plt.plot(range(0, episodes), outcomes,
             color="red")
    plt.xlabel("trial")
    plt.ylabel("survival time")
    plt.title("Q-Learning Training Outcomes")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    try:
        play()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        # plot()
        print("Stopping simulation...")
