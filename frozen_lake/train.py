# https://gymnasium.farama.org/environments/toy_text/frozen_lake/
# https://medium.com/data-science/q-learning-for-beginners-2837b777741
import time
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Settings
filename = "frozen_lake/qtable.txt"
load = True
slippery = True
visual = True
speed = 50

# Model hyperparameters
episodes = 10000
learning_rate = 0.5
discount_factor = 0.9
epsilon_init = 0
epsilon_decay = 5.0/episodes

env = gym.make('FrozenLake-v1',
               render_mode="human" if visual else None, is_slippery=slippery)
if visual:
    env.metadata['render_fps'] = speed
if load:
    qtable = np.loadtxt(filename, dtype=np.float16)
else:
    qtable = np.zeros(
        (env.observation_space.n, env.action_space.n), dtype=np.float16)
outcomes = np.zeros((episodes))
print("Initial", qtable)


def train():
    epsilon = epsilon_init
    for i in range(episodes):
        state, _ = env.reset()
        while True:
            # pick an action
            explore = np.random.random() < epsilon or abs(
                np.max(qtable[state])) < 1e-9
            if explore:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state])

            # perform the action
            new_state, reward, done, _, _ = env.step(action)

            # update qtable
            qtable[state, action] = (1-learning_rate)*qtable[state, action] + \
                learning_rate*(reward + discount_factor *
                               np.max(qtable[new_state]))

            state = new_state
            if visual:
                env.render()
            if done:
                outcomes[i] = reward
                # Update epsilon
                epsilon = max(epsilon - epsilon_decay, 0)
                break


def plot():
    # Plot outcomes
    window_size = 100
    moving_avg = np.convolve(outcomes, np.ones(
        window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, episodes), moving_avg,
             color="red")
    plt.xlabel("trial")
    plt.ylabel("success rate")
    plt.title("Q-Learning Training Outcomes (Frozen Lake)")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    try:
        train()
        plot()
    except KeyboardInterrupt:
        pass
    finally:
        print("Saving Qtable and stopping training...")
        print("Final", qtable)
        np.savetxt(filename, qtable)
