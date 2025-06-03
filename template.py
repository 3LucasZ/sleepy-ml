import time
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Model hyperparameters
episodes = 10000

env = gym.make('LunarLander-v3',
               render_mode="human")


def train():
    for i in range(episodes):
        state, _ = env.reset()
        totReward = 0
        while True:
            # pick an action
            action = env.action_space.sample()

            # perform the action
            new_state, reward, done, _, _ = env.step(action)
            totReward += reward
            state = new_state
            if done:
                print("Episode finished. Score:", totReward)
                break


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        pass
    finally:
        pass
