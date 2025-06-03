from collections import deque, namedtuple
import random
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import torch
from utils import *


# Settings
episodes = 50
model = -1

env = gym.make('LunarLander-v3',
               render_mode="human")


policy = StochasticPolicy(env.observation_space.shape[0], env.action_space.n)


buffer = torch.load(
    best_model_filepath(model) if model > -1 else model_filepath, weights_only=True)
policy.load_state_dict(buffer)

outcomes = []
maxSteps = 1000


def play():
    for i in range(episodes):
        state, _ = env.reset()
        rewardTot = 0
        steps = 0
        while True:
            # pick an action
            action, _ = policy.selectAction(state)

            # perform the action
            new_state, reward, done, _, _ = env.step(action)
            print(reward)
            state = new_state
            rewardTot += reward
            steps += 1

            if done or steps > maxSteps:
                outcomes.append(rewardTot)
                break


def plot():
    # Plot outcomes
    plt.plot(range(0, len(outcomes)), outcomes,
             color="red")
    plt.xlabel("trial")
    plt.ylabel("survival time")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    try:
        play()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        plot()
        print("Stopping simulation")
