from state import State
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
        # predicted Q value of each state
        self.predicted_v = {}
        # set of visited states states 
        self.visited = set()
        # some paramter
        self.c_puct = 1.0


    def search(self, s : State, max_depth = 10):
        # print("Search : ")
        # print(s.toStr())
        if s.gameEnded(): return s.gameReward()

        if s not in self.visited:
            self.visited.add(s)
            pi, v = self.nnet.predict(s)
            self.predicted_v[s] = v
            self.P[s] = pi
            self.N[s] = np.zeros(len(pi))
            self.Q[s] = np.zeros(len(pi))
            return v

        if max_depth == 0:
            print(f"max depth reached!, a heuristic value of this state {self.predicted_v[s]}")
            return self.predicted_v[s]
      
        max_u, best_a = -np.inf, None
        for a in s.getValidActions():
            u = self.Q[s][a] + self.c_puct * self.P[s][a] * np.sqrt(np.sum(self.N[s])) / (1 + self.N[s][a])
            if u > max_u:
                max_u = u
                best_a = a
        a = best_a
        
        sp, player_changed = s.nextState(a)
        v = self.search(sp, max_depth - 1)
        if player_changed:
            v = -v

        self.Q[s][a] = (self.N[s][a] * self.Q[s][a] + v) / (self.N[s][a] + 1)
        self.N[s][a] += 1
        return v

    # improved policy
    def pi(self, s : State):
        return self.N[s] / np.sum(self.N[s])

