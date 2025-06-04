from matplotlib import pyplot as plt
import torch
from utils import *

policy = StochasticPolicy(8*8*13, 8*8*8*8)
opponentWindow = 4
opponent = None
maxSteps = 200
outcomes = []
discount_factor = 0.99
adam_learning_rate = 0.001

env = Environment()
optimizer = torch.optim.Adam(policy.parameters(), lr=adam_learning_rate)

def play():
    state = env.create(opponent)
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

        rewards.append(reward)
        log_probs.append(log_prob)
        rewardTot += reward
        state = new_state
        steps += 1

        if done or steps > maxSteps:
            break
    print("Episode finished. Score:", rewardTot)
    outcomes.append(rewardTot)

    ### Optimize model ###
    # Discounted trajectory rewards
    trajRewards = torch.tensor(rewards)
    for i in range(steps-2, -1, -1):
        trajRewards[i] += discount_factor*trajRewards[i+1]

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
    plt.plot(range(window_size-1, len(outcomes)), moving_avg,
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
