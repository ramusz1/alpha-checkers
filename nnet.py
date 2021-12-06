import numpy as np
from state import State

class NNet:

    def __init__(self, action_size):
        self.action_size = action_size

    def predict(self, state : State):
        # uniform policy
        p = np.full(self.action_size, 1.0 / self.action_size)
        v = 1
        return p, v
     
    def train(self, examples):
        return self
