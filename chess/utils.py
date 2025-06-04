import os
import numpy as np
import torch

class StochasticPolicy(torch.nn.Module):
    def __init__(self, n_states, n_actions):
        super(StochasticPolicy, self).__init__()
        self.fc1 = torch.nn.Linear(n_states, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return torch.softmax(x, dim=1)

    def selectAction(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        actionProbs = self.forward(state)
        m = torch.distributions.Categorical(actionProbs)
        action = m.sample()
        return action.item(), m.log_prob(action)

class Environment():
    def __init__(self):
        self.opponent: StochasticPolicy = None
        self.violationValue = -1e6
        self.pieceValues = [1, 3, 3, 5, 9, 1e6]

    def create(self, opponent: StochasticPolicy):
        self.opponent = opponent
        return createInitialState()

    def step(self, state, action):
        # perform the move
        # if invalid => punish and done
        # if piece captured => reward
        # opponent moves
        action, _ = self.opponent.selectAction(state)
        # if invalid => done
        # if piece captured => punish
        # reward
        pass


def createInitialState():
    board2d = np.zeros((8, 8), dtype=int)
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
    board2d[7] = [ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK]
    board2d[6] = [PAWN] * 8
    board2d[0] = [13-ROOK, 13-KNIGHT, 13-BISHOP, 13-QUEEN, 13-KING, 13-BISHOP, 13-KNIGHT, 13-ROOK]
    board2d[1] = [13-PAWN] * 8
    print(board2d)

    board3d = np.zeros((8,8,13), dtype=int)
    for i in range(8):
        for j in range(8):
            piece = board2d[i, j]
            board3d[i, j, piece] = 1
    return board3d

def actionIdToCoords(n):
    if not 0 <= n < 4096:
        raise ValueError("Number must be between 0 and 4095")
    w = n // 512
    x = (n % 512) // 64
    y = (n % 64) // 8
    z = n % 8
    return (w, x, y, z)

dir_path = os.path.dirname(os.path.realpath(__file__))
model_filepath = os.path.join(dir_path, "model.pth")


def best_model_filepath(num):
    return os.path.join(dir_path, f"best_model_{num}.pth")

if __name__ == "__main__":
    board3d = createInitialState()
    print("Layer 0 (Empty squares):")
    print(board3d[:, :, 0])

    print("Layer 1 (White pawns):")
    print(board3d[:, :, 1])

    print(actionIdToCoords(50), actionIdToCoords(100))