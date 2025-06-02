# https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# https://gymnasium.farama.org/environments/classic_control/cart_pole/
# actions: 0 = push cart left, 1 = push cart right
# states: [cart position, cart velocity, pole angle, pole angular velocity] continuous over R
from collections import deque, namedtuple
import random
import time
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import torch
from utils import *


# Model hyperparameters
adam_learning_rate = 0.001
discount_factor = 0.99
episodes = 250
end_penalty = -25
epsilon_init = 1
epsilon_decay = 2.0/episodes
target_update_freq = 500
memory_size = 10000
batch_size = 64
device = torch.device("cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', ))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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
               render_mode=None)
policy_net = DQN(
    env.observation_space.shape[0],  env.action_space.n).to(device)
target_net = DQN(
    env.observation_space.shape[0],  env.action_space.n).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), lr=adam_learning_rate)
memory = ReplayMemory(memory_size)
outcomes = np.zeros((episodes))


def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = policy_net(state)
        return torch.argmax(q_values).item()


def optimize_model():
    if len(memory) < batch_size:
        return
    batch = memory.sample(batch_size)
    state_batch, action_batch, reward_batch, next_state_batch = zip(
        *batch)
    state_batch = torch.FloatTensor(np.array(state_batch)).to(device)
    action_batch = torch.LongTensor(
        np.array(action_batch)).unsqueeze(1).to(device)
    reward_batch = torch.FloatTensor(np.array(reward_batch)).to(device)
    next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(device)

    predicted_q_values = policy_net(
        state_batch).gather(1, action_batch).squeeze()
    max_next_q_values = target_net(next_state_batch).max(1)[0]
    target_q_values = reward_batch + discount_factor * max_next_q_values
    loss = torch.nn.MSELoss()(predicted_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train():
    epsilon = epsilon_init
    steps = 0
    n_target_updates = 0
    n_optimizations = 0
    for i in range(episodes):
        state, _ = env.reset()
        rewardTot = 0
        while True:
            steps += 1
            # pick an action
            action = select_action(state, epsilon)

            # perform the action
            new_state, reward, done, _, _ = env.step(action)
            if done:
                memory.push(state, action, end_penalty, new_state)
            else:
                memory.push(state, action, reward, new_state)
            state = new_state
            rewardTot += reward

            optimize_model()
            n_optimizations += 1
            if steps % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())
                n_target_updates += 1

            if done:
                outcomes[i] = rewardTot
                # Update epsilon
                epsilon = max(epsilon - epsilon_decay, 0)
                break
    print("n_target_updates", n_target_updates,
          "n_optimizations", n_optimizations)


def plot():
    # Plot outcomes
    window_size = 10
    moving_avg = np.convolve(outcomes, np.ones(
        window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, episodes), moving_avg,
             color="red")
    plt.xlabel("trial")
    plt.ylabel("survival time")
    plt.title("Q-Learning Training Outcomes")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    try:
        ti = time.time()
        train()
        print(time.time() - ti, "time elapsed")
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        plot()
        print("Saving model and stopping training...")
        torch.save(policy_net.state_dict(), model_filepath)
