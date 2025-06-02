import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
from utils import *

# Settings
speed = 50


env = gym.make('CartPole-v1',
               render_mode="human")
env.metadata['render_fps'] = speed
qtable = np.load(filename)


def play():
    while True:
        state, _ = env.reset()
        state = discretizeState(state)
        rewardTot = 0
        while True:
            # pick an action
            action = np.argmax(qtable[state])

            # perform the action
            new_state, reward, done, _, _ = env.step(action)
            rewardTot += reward
            new_state = discretizeState(new_state)

            state = new_state
            if done:
                print(rewardTot)
                break


if __name__ == "__main__":
    try:
        play()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
