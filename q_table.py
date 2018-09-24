# -*- coding: utf-8 -*-
import random
from collections import deque

import gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

EPISODES = 1000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.batch_size = 32
        self.model = self._build_model()
        self.qtable = np.zeros((state_size[0], action_size))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size[0], activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if (np.random.uniform() < self.epsilon):
            return np.random.random_integers(0, self.action_size - 1)
        q = self.qtable[state, :]
        max_q = np.argmax(q)
        return max_q

    def train_data(self):
        mini_batch = random.sample(self.memory, self.batch_size)
        states = np.zeros((self.batch_size, 1), dtype="int16")
        next_states = np.zeros((self.batch_size, 1), dtype="int16")
        rewards = np.zeros((self.batch_size,), dtype="int16")
        actions = np.zeros((self.batch_size,), dtype="int16")

        for i in np.arange(self.batch_size):
            states[i] = int(mini_batch[i][0])
            next_states[i] = int(mini_batch[i][3])
            rewards[i] = int(mini_batch[i][2])
            actions[i] = int(mini_batch[i][1])

        return states, next_states, rewards, actions

    def replay(self):
        states, next_states, rewards, actions = self.train_data()
        for i in range(len(states)):
            current_q = self.qtable[states[i][0], actions[i]]
            next_q = np.amax(self.qtable[next_states[i][0], :])
            reward = rewards[i]
            print("state: %.f current_q: %.5f next_q: %.5f reward: %.5f learning_rate: %.5f gamma:%.5f" % (
            states[i][0], current_q, next_q, reward, self.learning_rate, self.gamma))
            print(self.qtable[states[i][0], :])
            self.qtable[states[i][0], actions[i]] = current_q + self.learning_rate * (
                        reward + self.gamma * next_q - current_q)
            print(self.qtable[states[i][0], :])

        print(self.qtable)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def stack_states(self, stacked_states, state, is_new_episode):

        if (is_new_episode):
            stacked_frames = deque([np.zeros((INPUT_SHAPE[0:2]), dtype=np.int) for i in range(INPUT_SHAPE[-1])],
                                   maxlen=4)


if __name__ == "__main__":
    env = gym.make('FrozenLake-v0')
    state_size = (env.observation_space.n,)
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32
    replay_start_size = 50

    for e in range(EPISODES):
        state = env.reset()
        #        state = np.reshape(state, [1, state_size[0]])
        for time in range(1000):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # reward = reward if not done else -10
            # next_state = np.reshape(next_state, [1, state_size[0]])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > replay_start_size:
                agent.replay()
