# https://gymnasium.farama.org/environments/box2d/lunar_lander/
import time
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import *

# Model hyperparameters
episodes = 2000
discount_factor = 1.0
adam_learning_rate = 0.0002


env = gym.make('LunarLander-v3', render_mode=None)
policy = StochasticPolicy(env.observation_space.shape[0], env.action_space.n)
optimizer = torch.optim.Adam(policy.parameters(), lr=adam_learning_rate)
outcomes = np.zeros((episodes))
maxSteps = 1000
maxFails = 100


def train():
    fails = 0
    for episode in range(episodes):
        state, _ = env.reset()
        rewardTot = 0

        ### Generate an episode ###
        rewards = []
        log_probs = []
        steps = 0
        while True:
            # pick an action
            action, log_prob = policy.selectAction(state)

            # perform the action
            new_state, reward, done, _, _ = env.step(action)
            if reward > 0:
                reward = reward*3-2
            rewards.append(reward)
            log_probs.append(log_prob)
            rewardTot += reward
            state = new_state
            steps += 1

            if done or steps > maxSteps:
                break
        print("Episode finished. Score:", rewardTot)
        outcomes[episode] = rewardTot
        if (steps > maxSteps):
            print("Max steps reached.")
            fails += 1
            if (fails > maxFails):
                break
            continue

        ### Optimize model ###
        # Discounted trajectory rewards
        trajRewards = torch.tensor(rewards)
        for i in range(steps-2, -1, -1):
            trajRewards[i] += discount_factor*trajRewards[i+1]
        # Normalize to stabilize training
        # trajRewards = (trajRewards - trajRewards.mean()) / \
        #     (trajRewards.std() + 1e-8)

        policy_loss = torch.mean(torch.stack(
            [trajRewards[i] * -log_probs[i] for i in range(steps)]))
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()


def plot():
    # Plot outcomes
    window_size = 5
    moving_avg = np.convolve(outcomes, np.ones(
        window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, episodes), moving_avg,
             color="red")
    plt.xlabel("trial")
    plt.ylabel("reward")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        plot()
        print("Saving model")
        torch.save(policy.state_dict(), model_filepath)
