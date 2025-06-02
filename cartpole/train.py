# https://gymnasium.farama.org/environments/classic_control/cart_pole/
# actions: 0 = push cart left, 1 = push cart right
# states: [cart position, cart velocity, pole angle, pole angular velocity] continuous over R
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
from utils import *

# Model hyperparameters
episodes = 3000
learning_rate = 0.1
discount_factor = 0.99
epsilon_init = 1
epsilon_decay = 2.0/episodes


env = gym.make('CartPole-v1',
               render_mode=None)
qtable = np.zeros(
    (len(pos_space)+1, len(d_pos_space)+1, len(ang_space)+1, len(d_ang_space)+1, env.action_space.n), dtype=np.float16)
outcomes = np.zeros((episodes))


def train():
    epsilon = epsilon_init
    for i in range(episodes):
        state, _ = env.reset()
        state = discretizeState(state)
        rewardTot = 0
        while True:
            # pick an action
            explore = np.random.random() < epsilon or abs(
                np.max(qtable[state]) - np.min(qtable[state])) < 1e-9
            if explore:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state])

            # perform the action
            new_state, reward, done, _, _ = env.step(action)
            rewardTot += reward
            new_state = discretizeState(new_state)

            # update qtable
            qtable[state+(action,)] = (1-learning_rate)*qtable[state+(action,)] + \
                learning_rate*(reward + discount_factor *
                               np.max(qtable[new_state]))
            state = new_state
            if done:
                outcomes[i] = rewardTot
                # Update epsilon
                epsilon = max(epsilon - epsilon_decay, 0)
                break


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
    plt.legend()
    plt.show()


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        plot()
        print("Saving Qtable and stopping training...")
        np.save(filename, qtable)
