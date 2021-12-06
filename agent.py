#based on https://web.stanford.edu/~surag/posts/alphazero.html
from state import State
from nnet import NNet
from pettingzoo.classic import checkers_v3
from pettingzoo.utils.env import AECEnv
import numpy as np

class MCTS:

    def __init__(self, nnet):
        self.nnet = nnet
        # number of times given state and action has been tested
        self.N = {}
        # policy in each state
        self.P = {}
        # Q value of each state
        self.Q = {}
        # set of visited states states 
        self.visited = set()
        # some paramter
        self.c_puct = 1.0


    def search(self, s : State):
        # print("Search : ")
        # print(s.toStr())
        if s.gameEnded(): return s.gameReward()

        if s not in self.visited:
            self.visited.add(s)
            pi, v = self.nnet.predict(s.getObservation())
            self.P[s] = pi
            self.N[s] = np.zeros(len(pi))
            self.Q[s] = np.zeros(len(pi))
            return v
      
        max_u, best_a = -np.inf, None
        for a in s.getValidActions():
            u = self.Q[s][a] + self.c_puct * self.P[s][a] * np.sqrt(np.sum(self.N[s])) / (1 + self.N[s][a])
            if u > max_u:
                max_u = u
                best_a = a
        a = best_a
        
        sp, player_changed = s.nextState(a)
        v = self.search(sp)
        if player_changed:
            v = -v

        self.Q[s][a] = (self.N[s][a] * self.Q[s][a] + v) / (self.N[s][a] + 1)
        self.N[s][a] += 1
        return v

    # improved policy
    def pi(self, s : State):
        return self.N[s] / np.sum(self.N[s])


def pit(new_nnet, nnet):
    frac_win = 1.0
    return frac_win

# training
def policyIterSP(env : AECEnv, num_iters = 1, num_eps = 1, frac_win_thresh = 0.6):
    # hard coded action space size
    nnet = NNet(256)
    examples = []
    for i in range(num_iters):
        for e in range(num_eps):
            examples += executeEpisode(env, nnet)           # collect examples from this game
        new_nnet = nnet.train(examples)
        frac_win = pit(new_nnet, nnet)                      # compare new net with previous net
        if frac_win > frac_win_thresh:
            nnet = new_nnet                                 # replace with new net
    return nnet


class Example:

    def __init__(self, state : State, pi, reward):
        self.state = state
        self.pi = pi
        self.reward = reward


def executeEpisode(env : AECEnv, nnet, num_mcts_sims = 3):
    examples = []
    env.reset()
    s = State(env)
    s.show(wait = True)
    mcts = MCTS(nnet)                                           # initialise search tree

    while True:
        for _ in range(num_mcts_sims):
            mcts.search(s)
        pi = mcts.pi(s)
        examples.append(Example(s, pi, None))              # rewards can not be determined yet
        a = np.random.choice(len(pi), p=pi)    # sample action from improved policy
        s, _ = s.nextState(a)
        s.show(wait = True)
        if s.gameEnded():
            examples = assignRewards(examples, s.gameReward(), s.currentAgent())
            return examples

def assignRewards(examples, reward, player_w_reward):
    for i, e in enumerate(examples):
        e.reward = reward if e.state.currentAgent() == player_w_reward else -reward

    return examples

if __name__ == "__main__":
    env = checkers_v3.env()
    policyIterSP(env)
