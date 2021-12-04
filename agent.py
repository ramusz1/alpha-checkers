#based on https://web.stanford.edu/~surag/posts/alphazero.html

class MCTS:

    def search(s, game, nnet):
        if game.gameEnded(s): return -game.gameReward(s)

        if s not in visited:
            visited.add(s)
            P[s], v = nnet.predict(s)
            return -v
      
        max_u, best_a = -float("inf"), -1
        for a in game.getValidActions(s):
            u = Q[s][a] + c_puct*P[s][a]*sqrt(sum(N[s]))/(1+N[s][a])
            if u>max_u:
                max_u = u
                best_a = a
        a = best_a
        
        sp = game.nextState(s, a)
        v = search(sp, game, nnet)

        Q[s][a] = (N[s][a]*Q[s][a] + v)/(N[s][a]+1)
        N[s][a] += 1
        return -v


# training
def policyIterSP(game):
    nnet = initNNet()                                       # initialise random neural network
    examples = []
    for i in range(numIters):
        for e in range(numEps):
            examples += executeEpisode(game, nnet)          # collect examples from this game
        new_nnet = trainNNet(examples)
        frac_win = pit(new_nnet, nnet)                      # compare new net with previous net
        if frac_win > threshold:
            nnet = new_nnet                                 # replace with new net
    return nnet

def executeEpisode(game, nnet):
    examples = []
    s = game.startState()
    mcts = MCTS()                                           # initialise search tree

    while True:
        for _ in range(numMCTSSims):
            mcts.search(s, game, nnet)
        examples.append([s, mcts.pi(s), None])              # rewards can not be determined yet
        a = random.choice(len(mcts.pi(s)), p=mcts.pi(s))    # sample action from improved policy
        s = game.nextState(s,a)
        if game.gameEnded(s):
            examples = assignRewards(examples, game.gameReward(s))
            return examples
