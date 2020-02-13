import logging
import keras.models
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense


class DQLAgent(object):
    def __init__(
            self, state_size=-1, action_size=-1,
            max_steps=200, gamma=1.0, epsilon=.001, learning_rate=0.01, filename="weights.h5"):
        self.state_size = state_size # 6
        self.action_size = action_size
        self.max_steps = max_steps
        self.memory = deque(maxlen=2000)
        self.gamma = gamma   # discount rate
        self.epsilon = epsilon  # exploration rate
        self.learning_rate = learning_rate  # learning_rate
        if self.state_size > 0 and self.action_size > 0:
            if filename: self.load(filename)
            else: self.model = self.build_model()

        self.count = 0

    def build_model(self):
        """Neural Net for Deep-Q learning Model."""
        model = Sequential()

        # TODO(students): !!!!!!!!! IMPLEMENT THIS !!!!!!!!!!!!!!  """
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(48, activation='relu'))
        # model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, decay=.01))
        return model

    def updateEpsilon(self):
        """This function change the value of self.epsilon to deal with the
        exploration-exploitation tradeoff as time goes"""

        # TODO(students): !!!!!!!!! IMPLEMENT THIS !!!!!!!!!!!!!!  """
        # It should change (or not) the value of self.epsilon
        # self.epsilon -= (0.9 - 0.1) / self.NUM_EPISODES
        # self.epsilon = 0.8*abs(np.sin(self.count/100))
        if self.epsilon > .01:
            self.epsilon *= 0.9995


    def save(self, output: str):
        self.model.save(output)

    def load(self, filename):
        if os.path.isfile(filename):
            self.model = keras.models.load_model(filename)
            self.state_size = self.model.layers[0].input_shape[1]
            self.action_size = self.model.layers[-1].output.shape[1]
            return True
        else:
            logging.error('no such file {}'.format(filename))
            return False

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, env, greedy=True):
        # !!!!  TODO(students): implements an epsilon greedy policy here   !!!!
        # Make sure that if greedy is True, the policy should be greedy
        # As it is the policy is just plain greedy....

        if greedy or np.random.random() > self.epsilon:
            return np.argmax(self.model.predict(state))
        else:
            return random.randrange(len(env.actions))


    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        self.updateEpsilon()

    def setTitle(self, env, train, name, num_steps, returns):
        h = name
        if train:
            h = 'Iter {} ($\epsilon$={:.2f})'.format(self.count, self.epsilon)
        end = '\nreturn {:.2f}'.format(returns) if train else ''

        env.mayAddTitle('{}\nsteps: {} | {}{}'.format(
            h, num_steps, env.circuit.debug(), end))

    def run_once(self, env, train=True, greedy=False, name=''):
        self.count += 1
        state = env.reset()
        state = np.reshape(state, [1, self.state_size])
        returns = 0
        num_steps = 0
        while num_steps < self.max_steps:
            num_steps += 1
            action = self.act(state, env, greedy=greedy)
            next_state, reward, done = env.step(action, greedy)
            next_state = np.reshape(next_state, [1, self.state_size])

            if train:
                self.remember(state, action, reward, next_state, done)

            returns = returns * self.gamma + reward
            state = next_state
            if done:
                return returns, num_steps

            self.setTitle(env, train, name, num_steps, returns)

        return returns, num_steps

    def train(
            self, env, episodes, minibatch, output='weights.h5', graph=False):
        best_r = 0.0
        # R, E, GR= [0]*700, [0]*700, [0]*700
        R, E, GR= [], [], []
        if graph:
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            plt.ion()
        self.NUM_EPISODES = episodes
        self.flag = False
        for e in range(episodes):
            self.flag = not self.flag
            r, _ = self.run_once(env, train=True, greedy=self.flag)
            if self.flag:
                if r > best_r:
                    print("SAVED")
                    self.save(output)
                GR.append(r)
                # GR.pop(0)
            else:
                R.append(r)
                E.append(self.epsilon)
                # R.pop(0)
                # E.pop(0)
            if r > best_r: best_r = r
            print("episode: {}/{}, return: {:2.2f}/{:2.2f}, e: {}".format(
                e, episodes, r, best_r, self.epsilon))

            if len(self.memory) > minibatch:
                self.replay(minibatch)
                # self.save(output)
            if graph:
                plt.clf()
                ax1.plot(R, color='red')
                ax1.plot(GR, color='green')
                ax2.plot(E, color='blue')
                ax2.set_ylim([0,1])
                plt.axes(ax1)
                plt.axes(ax2)
                plt.draw()
                plt.pause(0.0001)
        # plt.show()
        # Finally runs a greedy one
        r, n = self.run_once(env, train=False, greedy=True)
        self.save(output)
        print("Greedy return: {} in {} steps".format(r, n))
