#based on https://web.stanford.edu/~surag/posts/alphazero.html
from state import State
from nnet import NNet, RandomPlayer
from training_example import TrainingExample
from mcts import MCTS

from pettingzoo.classic import checkers_v3
from pettingzoo.utils.env import AECEnv
from copy import deepcopy
import numpy as np

def pit(new_nnet : NNet, nnet : NNet, games_played = 40):
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
def policyIterSP(env : AECEnv, num_iters = 10, num_eps = 10,  num_mcts_sims=25, frac_win_thresh = 0.55):
    # hard coded action space size
    nnet = NNet(256)
    frac_win = pit(nnet, RandomPlayer())                              # compare new net with a random player
    print("frac_wins against a random player", frac_win)
    examples = []
    for i in range(num_iters):
        for e in range(num_eps):
            examples += executeSelfPlayEpisode(env, nnet, num_mcts_sims)    # collect examples from this game
            print("episode done")
        new_nnet = nnet.train(examples)
        frac_win = pit(new_nnet, nnet)                                # compare new net with previous net
        print("frac_win", frac_win)
        if frac_win > frac_win_thresh:
            print("new net is better!")
            nnet = new_nnet                                           # replace with new net
            frac_win = pit(nnet, RandomPlayer())                      # compare new net with a random player
            print("frac_wins against a random player", frac_win)
        examples = random.sample(examples, len(examples) // 2)        # discard half of the examples
    return nnet

def executeSelfPlayEpisode(env : AECEnv, nnet, num_mcts_sims = 3):
    examples = []
    env.reset()
    s = State(env)
    # s.show(wait = False)
    mcts = MCTS(nnet, num_mcts_sims)

    while True:
        mcts.search(s)
        pi = mcts.pi(s)
        examples.append(TrainingExample(deepcopy(s), pi, None))  # rewards can not be determined yet
        a = np.random.choice(len(pi), p=pi)                      # sample action from improved policy
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
    import argparse
    parser = argparse.ArgumentParser(description="train checkers ai")
    parser.add_argument("--test", action="store_true", help="run short test")
    args = parser.parse_args()
    if args.test:
        print(
            '''
####################
running test version
####################
            '''
        )
        env = checkers_v3.env()
        nnet = policyIterSP(env, num_iters=1, num_eps=1, num_mcts_sims=3)
    else:
        env = checkers_v3.env()
        nnet = policyIterSP(env, num_iters=8, num_eps=50, num_mcts_sims=25)

