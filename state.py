from pettingzoo.utils.env import AECEnv
from copy import deepcopy
import numpy as np

# Game env wrapper for MCTS search
class State:

    def __init__(self, env : AECEnv):
        self.env = env

    def gameEnded(self):
        _, _, done, _ = self.env.last()
        return done

    def gameReward(self):
        _, reward, _, _ = self.env.last()
        return reward

    def getActionMask(self):
        observation, _, _, _ = self.env.last()
        return observation["action_mask"]

    def getValidActions(self):
        return np.flatnonzero(self.getActionMask())

    def nextState(self, action):
        new_env = deepcopy(self.env)
        new_env.step(action)
        player_changed = self.env.agent_selection != new_env.agent_selection
        return State(new_env), player_changed

    def getObservation(self):
        return self.env.observe(self.currentAgent())["observation"]

    def currentAgent(self):
        return self.env.agent_selection

    def show(self, wait=False):
        self.env.render()
        if wait:
            input("press any key to continue")


    def __eq__(self, x):
        if not isinstance(x, State):
            return False
        # this should be enough
        same_agent = self.env.agent_selection == x.env.agent_selection
        observations_match = (self.getObservation() == x.getObservation()).all()
        return same_agent and observations_match

    def toStr(self):
        o = self.getObservation()
        # reduce dimensions from 3 to 2
        o = np.sum(o, axis = 2) * (np.argmax(o, axis = 2) + 1)
        return str(o)

    def __hash__(self):
        return hash(self.toStr())
