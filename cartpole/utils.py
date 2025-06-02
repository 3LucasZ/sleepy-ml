# Continuous => discrete
import numpy as np

filename = "cartpole/qtable.npy"

ticks = 15
pos_space = np.linspace(-2.4, 2.4, ticks)
d_pos_space = np.linspace(-4, 4, ticks)
ang_space = np.linspace(-.2095, .2095, ticks)
d_ang_space = np.linspace(-4, 4, ticks)


def discretizeState(state):
    state_pos = np.digitize(state[0], pos_space)
    state_d_pos = np.digitize(state[1], pos_space)
    state_ang = np.digitize(state[2], pos_space)
    state_d_ang = np.digitize(state[3], pos_space)
    return (state_pos, state_d_pos, state_ang, state_d_ang)
