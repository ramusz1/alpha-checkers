from state import State
import numpy as np

class MCTSNode:

    def __init__ (self, p, q):
        """
        Parameters
        ----------
        p : policy in this state
        q : q value of this state
        """
        self.p = p
        self.q = q
        # n[a] : number of times and action has been performed from this state
        self.n = np.zeros(len(p))
        # q_a : q values of states following performing an action a
        self.q_a = np.zeros(len(p))


class MCTS:

    def __init__(self, nnet, num_mcts_sims, max_depth = 10):
        self.nnet = nnet
        self.nodes = {}
        self.c_puct = 1.0
        self.num_mcts_sims = num_mcts_sims
        self.max_depth = max_depth
        
    def search(self, s : State):
        for _ in range(self.num_mcts_sims):
            self._search(s, self.max_depth)

    def _search(self, s : State, max_depth):
        if s.gameEnded(): return s.gameReward()

        if s not in self.nodes:
            p, v = self.nnet.predict(s)
            self.nodes[s] = MCTSNode(p, v)
            return v

        node = self.nodes[s]

        if max_depth == 0:
            # max depth reached, returning a heuristic value of this state
            return node.q
      
        # upper confidence bound
        ucb = node.q_a + self.c_puct * node.p * np.sqrt(np.sum(node.n)) / (1 + node.n)
        ucb[s.getActionMask() == 0] = -np.inf
        # choose best action based on ucb
        a = np.argmax(ucb)
        
        sp, player_changed = s.nextState(a)
        v = self._search(sp, max_depth - 1)
        if player_changed:
            v = -v

        node.q_a[a] = (node.n[a] * node.q_a[a] + v) / (node.n[a] + 1)
        node.n[a] += 1
        return v

    # improved policy
    def pi(self, s : State):
        node = self.nodes[s]
        n_sum = np.sum(node.n)
        if n_sum == 0:
            return node.p

        return node.n / n_sum


if __name__ == "__main__":
    # mcts = MCTS(nnet, 2)
    # mcts.search(state)
    # mcts.pi(state).shape
    from nnet import NNet
    import timeit

    nnet = NNet(256)
    mcts = MCTS(nnet, 30)
    # timeit.timeit("mcts.search(state)", globals=globals())

    # timeit -r 2 -n 5 mcts.search(state)
    # colab times:
    # old version time: 5 loops, best of 2: 4.6 s per loop
    # current time: 5 loops, best of 2: 1.54 s per loop

