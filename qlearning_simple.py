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
        self.gamma = 0.99  # discount rate
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

        model.add(Dense(24, input_shape=(self.batch_size, self.state_size[0]), activation='relu'))
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
        states = np.zeros((self.batch_size,) + self.state_size, dtype="int16")
        next_states = np.zeros((self.batch_size,) + self.state_size, dtype="int16")
        rewards = np.zeros((self.batch_size,), dtype="int16")
        actions = np.zeros((self.batch_size,), dtype="int16")
        done = np.zeros((self.batch_size,), dtype="int16")

        for i in np.arange(self.batch_size):
            states[i] = mini_batch[i][0]
            next_states[i] = mini_batch[i][3]
            rewards[i] = mini_batch[i][2]
            actions[i] = mini_batch[i][1]
            done[i] = mini_batch[i][4]

        return states, next_states, rewards, actions, done

    def replay(self):
        states, next_states, rewards, actions, done = self.train_data()
        # next_states = np.reshape(next_states, (1,) + next_states.shape)
        next_q_values = self.model.predict(next_states, batch_size=self.batch_size, verbose=0)
        target = rewards + self.gamma * np.amax(next_q_values, axis=3)
        target[done] = rewards

        states = np.reshape(states, (1,) + states.shape)
        self.model.fit(states, target)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape
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
