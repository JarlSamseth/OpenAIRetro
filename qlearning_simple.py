# -*- coding: utf-8 -*-
import random
from collections import deque

import gym
import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam

from TfSummary import TfSummary

EPISODES = 1000


class DQNAgent:
    def __init__(self, state_size, action_size, action_space):
        self.state_size = state_size
        self.action_space = action_space
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.001
        self.batch_size = 32
        self.input_shape = self.state_size
        self.model = self._build_model()


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input = Input(shape=self.input_shape, name='frames')
        x = Dense(units=50, activation='relu')(input)
        x = Dense(units=50, activation='relu')(x)
        output = Dense(units=self.action_size, activation='linear')(x)
        model = Model(input=input, output=output)
        model.compile(optimizer=Adam(lr=self.learning_rate), loss="mse")
        print(model.summary())

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if (np.random.uniform() < self.epsilon):
            return self.action_space.sample()

        state = np.reshape(state, (1,) + state.shape)

        q_values = self.model.predict(state, verbose=0)
        max_q = np.argmax(q_values)
        return max_q

    def train_data(self):
        mini_batch = random.sample(self.memory, self.batch_size)
        states = np.zeros((self.batch_size,) + self.state_size, dtype="int16")
        next_states = np.zeros((self.batch_size,) + self.state_size, dtype="int16")
        rewards = np.zeros((self.batch_size,), dtype="int16")
        actions = np.zeros((self.batch_size,), dtype="int16")
        done = np.zeros((self.batch_size,), dtype="bool")

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
        # states = np.reshape(states, (1,) + states.shape)
        next_q_values = self.model.predict(next_states, batch_size=self.batch_size, verbose=0)
        q_values = self.model.predict(states, batch_size=self.batch_size, verbose=0)

        for i in range(self.batch_size):
            if done[i]:
                q_values[i][actions[i]] = rewards[i]
            else:
                q_values[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_q_values[i])

        self.model.fit(states, q_values, verbose=0)
        if (self.epsilon < self.epsilon_min):
            self.epsilon = self.epsilon_min
        else:
            self.epsilon = self.epsilon * self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, env.action_space)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32
    replay_start_size = 100
    summary = TfSummary("MountainCar", ["Total_Reward/Episode", "Steps/Episode"])

    for e in range(EPISODES):
        state = env.reset()
        #        state = np.reshape(state, [1, state_size[0]])
        total_reward = 0
        episode_step = 0
        for time in range(1000):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # reward = reward if not done else -10
            # next_state = np.reshape(next_state, [1, state_size[0]])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            episode_step += 1
            if done:
                print("episode: {}/{}, tot_reward: {}, step: {} score: {}, e: {:.2}"
                      .format(e, EPISODES, total_reward, episode_step, time, agent.epsilon))
                break
            if len(agent.memory) > replay_start_size:
                agent.replay()

        summary.add_to_summary({"Total_Reward/Episode": total_reward, "Steps/Episode": episode_step}, e)
