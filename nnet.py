import numpy as np
from state import State
from training_example import TrainingExample
from typing import List
from random import shuffle
from tensorflow.keras import layers, Model, Input, metrics
import numpy as np

class NNet:

    def __init__(self, action_size):
        x = Input(shape=(8,8,4))
        y = layers.Conv2D(16, 3, activation='relu')(x)
        y = layers.Conv2D(16, 3, activation='relu')(y)
        y = layers.Conv2D(16, 3, strides=2, activation='relu')(y)
        y = layers.Flatten()(y)
        y = layers.Dropout(0.5)(y)
        p = layers.Dense(action_size, activation='softmax', name="pi")(y)
        v = layers.Dense(1, name="v")(y)
        self.nnet = Model(x, [p,v])
        print(self.nnet.summary())
        self.nnet.compile(
                optimizer='rmsprop',
                loss=["categorical_crossentropy","mean_squared_error"],
                metrics=[metrics.MeanSquaredError(), metrics.CategoricalCrossentropy()]
        )

    def predict(self, state : State):
        x = state.getObservation()
        x = np.expand_dims(x,0)
        p, v = self.nnet.predict(x, batch_size=1)

        p = p[0]
        m = state.getActionMask()
        p *= m
        # nnet zeroed all possible actions
        if np.sum(p) == 0:
            p = m / np.sum(m)
        else:
            p = p / np.sum(p) # renormalize
        return p, v[0][0]

    @staticmethod
    def _prepare_examples(examples: List[TrainingExample]):
        X = []
        pi = []
        v = []
        shuffle(examples)
        for e in examples:
            X.append(e.state.getObservation())
            pi.append(e.pi)
            v.append(e.reward)
        
        return np.array(X), [np.array(pi), np.array(v)]
     
    def train(self, examples):
        X, y = self._prepare_examples(examples)
        self.nnet.fit(X, y, batch_size=32)
        return self

if __name__ == "__main__":
    nnet = NNet(256)
    from pettingzoo.classic import checkers_v3
    env = checkers_v3.env()
    env.reset()
    state = State(env)
    nnet.predict(state)
