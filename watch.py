from pettingzoo.classic import checkers_v3
import numpy as np

def policy(observation, agent):
    # random policy
    possible_actions = np.flatnonzero(observation["action_mask"])
    if len(possible_actions) == 0:
        return None
    return np.random.choice(possible_actions)

if __name__ == "__main__":
    env = checkers_v3.env()
    env.reset()
    env.render()
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()
        action = policy(observation, agent)
        _ = input("press any key to continue")
        env.step(action)
        env.render()

