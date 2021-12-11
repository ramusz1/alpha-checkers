#based on https://web.stanford.edu/~surag/posts/alphazero.html
from state import State
from nnet import NNet
from training_example import TrainingExample
from mcts import MCTS

from pettingzoo.classic import checkers_v3
from pettingzoo.utils.env import AECEnv
from copy import deepcopy
import numpy as np

def pit(new_nnet : NNet, nnet : NNet, games_played = 10):
    new_nnet_tag = "player_0"
    nnet_tag = "player_1"
    wins = 0
    ties = 0

    for g in range(games_played):
        env = checkers_v3.env()
        env.reset()
        s = State(env)
        # swap players before each round
        new_nnet_tag, nnet_tag = nnet_tag, new_nnet_tag  
        agents = {new_nnet_tag : new_nnet, nnet_tag : nnet}

        while not s.gameEnded():
            agent = agents[s.currentAgent()]
            p, _ = agent.predict(s)
            action = np.random.choice(len(p), p=p)
            s.env.step(action)

        if s.gameReward() == 0:
            ties += 1
       
        if s.gameReward() == 1 and s.currentAgent() == new_nnet_tag:
            wins += 1
    
        if s.gameReward() == -1 and s.currentAgent() != new_nnet_tag:
            wins += 1
        
            
    frac_win = wins / (games_played - ties)
    return frac_win

# training
def policyIterSP(env : AECEnv, num_iters = 1, num_eps = 1, frac_win_thresh = 0.55):
    # hard coded action space size
    nnet = NNet(256)
    examples = []
    for i in range(num_iters):
        for e in range(num_eps):
            examples += executeSelfPlayEpisode(env, nnet)           # collect examples from this game
            print("episode done")
        new_nnet = nnet.train(examples)
        frac_win = pit(new_nnet, nnet)                      # compare new net with previous net
        print("frac_win", frac_win)
        if frac_win > frac_win_thresh:
            print("new net is better!")
            nnet = new_nnet                                 # replace with new net
    return nnet

def executeSelfPlayEpisode(env : AECEnv, nnet, num_mcts_sims = 2):
    examples = []
    env.reset()
    s = State(env)
    # s.show(wait = False)
    mcts = MCTS(nnet)                                           # initialise search tree

    while True:
        for _ in range(num_mcts_sims):
            mcts.search(s)
        pi = mcts.pi(s)
        examples.append(TrainingExample(deepcopy(s), pi, None))              # rewards can not be determined yet
        a = np.random.choice(len(pi), p=pi)    # sample action from improved policy
        s, _ = s.nextState(a)
        # s.show(wait = False)
        if s.gameEnded():
            examples = assignRewards(examples, s.gameReward(), s.currentAgent())
            return examples

def assignRewards(examples, reward, player_w_reward):
    for e in examples:
        e.reward = reward if e.state.currentAgent() == player_w_reward else -reward

    return examples

if __name__ == "__main__":
    env = checkers_v3.env()
    policyIterSP(env)
