import os

dir_path = os.path.dirname(os.path.realpath(__file__))
model_filepath = os.path.join(dir_path, "model.pth")


def best_model_filepath(num):
    return os.path.join(dir_path, f"best_model_{num}.pth")
