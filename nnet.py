import numpy as np
from state import State
from training_example import TrainingExample
from typing import List
from tensorflow.keras import layers, Model, Input, metrics, losses
import tensorflow as tf

def applyActionMaskToPolicy(p, action_mask):
    p_masked = p * action_mask
    # policy zeroed all possible actions
    if np.sum(p_masked) == 0:
        p_masked = m

    return p_masked / np.sum(p_masked) # renormalize


class NNet:

    def __init__(self, action_size):
        x = Input(shape=(8,8,4))
        y = layers.Conv2D(32, 3, activation='relu')(x)
        y = layers.Conv2D(32, 3, activation='relu')(y)
        y = layers.Flatten()(y)
        # y = layers.Dropout(0.5)(y)
        p = layers.Dense(action_size, activation='softmax', name="p")(y)
        v = layers.Dense(1, name="v")(y)
        self.nnet = Model(x, [p,v])
        print(self.nnet.summary())
        
        def entropyLoss(y_true, y_pred):
            return -y_true * tf.math.log(y_pred + 1e-10)
            
        self.nnet.compile(
                optimizer="adam",
                loss={"p": entropyLoss, "v":"mse"}
        )

    def predict(self, state : State):
        x = state.getObservation()
        x = np.expand_dims(x, 0)
        p, v = self.nnet.predict(x, batch_size=1)

        p = p[0]
        p = applyActionMaskToPolicy(p, state.getActionMask())
        return p, v[0][0]

    @staticmethod
    def _prepare_examples(examples: List[TrainingExample]):
        X = []
        pi = []
        v = []
        for e in examples:
            X.append(e.state.getObservation())
            pi.append(e.pi)
            v.append(e.reward)
        
        return np.array(X), [np.array(pi), np.array(v)]
     
    def train(self, examples):
        X, y = self._prepare_examples(examples)
        self.nnet.fit(X, y, batch_size=32, shuffle=True, epochs=3)
        return self

class RandomPlayer:
    
    def predict(self, state : State):
        p = np.random.uniform(256)
        return applyActionMaskToPolicy(p, state.getActionMask()), 0


if __name__ == "__main__":
    nnet = NNet(256)
    from pettingzoo.classic import checkers_v3
    env = checkers_v3.env()
    env.reset()
    env.render()
    state = State(env)
    nnet.predict(state)
    examples = [TrainingExample(state, np.full(256, 1.0 / 256), 1) for _ in range(32)]
    nnet.train(examples)

