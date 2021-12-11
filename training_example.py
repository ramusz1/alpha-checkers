from state import State

class TrainingExample:

    def __init__(self, state : State, pi, reward):
        self.state = state
        self.pi = pi
        self.reward = reward


