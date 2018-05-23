from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers import Conv2D, Dense, Flatten, LSTM, merge, BatchNormalization, MaxPooling2D, Input, AveragePooling2D, \
    Lambda, Merge, \
    Activation, Embedding
from keras.optimizers import Adam
import numpy as np
import random
import os
import logging as log
from collections import deque
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent
import pandas as pd
import pickle
import datetime

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

log.basicConfig(level=log.INFO)


class BaseAgent:

    def __init__(self, state_size, action_size):
        self.weight_backup = "cartpole_weight.h5"
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.exploration_rate = 1.0
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.memory = deque(maxlen=10000)
        self.randBinList = lambda n: [random.randint(0, 1) for b in range(1, n + 1)]
        self.model = None

    def __build_model(self):
        return None

    def load_model_from_file(self, backup_folder_name, model_name):
        log.info("Loading model and memory")
        if os.path.isfile(script_dir + "/" + backup_folder_name + "/" + model_name):
            self.model.load_weights(script_dir + "/" + backup_folder_name + "/" + model_name)
            self.memory = pickle.load(open(script_dir + "/" + backup_folder_name + "/" + "memory_file.pckl", "rb"))
            self.exploration_rate = pickle.load(
                open(script_dir + "/" + backup_folder_name + "/" + "exploartion_rate.pckl", "rb"))
        else:
            log.warning("Could not find model on path")
        log.info("Loading successful")

    def save_model(self, model_name):
        self.model.save(model_name)

    def train(self, state, reward_value, verbose=1):
        self.model.train_on_batch(state, reward_value)

    def predict(self, state):
        return self.model.predict(state)

    def calculate_target(self, reward, next_state):

        return reward + self.gamma * np.max(self.predict(next_state)[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save_model_and_memory(self, model_name, dir_name):
        log.info("Saving model and memory")
        if (not os.path.isdir(script_dir + "\\" + dir_name)):
            os.mkdir(script_dir + "\\" + dir_name, 0o777)
        pickle.dump(self.memory, open(script_dir + "/" + dir_name + "/memory_file.pckl", "wb"))
        pickle.dump(self.exploration_rate, open(script_dir + "/" + dir_name + "/exploartion_rate.pckl", "wb"))
        self.model.save(script_dir + "/" + dir_name + "/" + model_name)
        log.info("Saving successful")

    def act(self, state):
        #Buttons: ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
        if np.random.random() <= self.exploration_rate:
            return self.randBinList(self.action_size)
        q_values = self.model.predict(state)
        #q_act = np.zeros((1, 12))[0]
        #q_act[np.argmax(q_values[0])] = 1
        return q_values[0]

    def replay(self, sample_batch_size):

        if len(self.memory) < sample_batch_size:
            return

        sample_batch = self.get_sample_batch(sample_batch_size)
        targets = np.zeros((sample_batch_size, self.action_size))
        states=np.zeros((sample_batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        i = 0
        for state, action, reward, next_state, done in sample_batch:
            target=self.predict(state)
            #targets[i] = self.predict(state)
            #states[i] = state
            if done:
                #targets[i][np.array(list(map(int, action))) == 0]=0
                #targets[i][np.argmax(action)] = reward
                #target[0, np.array(action)==1] = reward
                Q_values = np.array([reward])
            else:
                #target = self.calculate_target(reward, next_state)
                #targets[i][np.array(list(map(int, action))) == 0]=0
                #targets[i][np.argmax(action)] = reward + self.gamma * np.max(self.predict(next_state)[0])
                Q_values = reward + self.gamma * np.max(self.predict(next_state), axis=1)
                #target[0, np.array(action)==1] = reward + self.gamma * np.argmax(self.predict(next_state)[0])

            i += 1
            state = np.reshape(state, (1,) + self.state_size)
            self.train(state, action*Q_values[:, None], verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

    def get_sample_batch(self, sample_batch_size):
        return random.sample(self.memory, sample_batch_size)


class DeepQAgent(BaseAgent):

    def __init__(self, state_size, action_size):
        BaseAgent.__init__(self, state_size, action_size)
        self.__build_model()

    def __build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, 8, 8, subsample=(4, 4), activation='relu', input_shape=(self.state_size)))
        self.model.add(Conv2D(64, 4, 4, subsample=(2, 2), activation='relu'))
        self.model.add(Conv2D(64, 3, 3, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(output_dim=512, activation='relu'))
        self.model.add(Dense(output_dim=self.action_size, activation='linear'))

        adam = Adam(lr=self.learning_rate)
        self.model.compile(loss='mse', optimizer=adam)


class DeepQLstmAgent(BaseAgent):

    def __init__(self, state_size, action_size):
        BaseAgent.__init__(self, state_size, action_size)

        self.value_size = 1
        self.__build_model()

    def __build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, 8, 8, subsample=(4, 4), input_shape=(self.state_size)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, 4, 4, subsample=(2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, 3, 3))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(units=64))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dense(units=32))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dense(output_dim=self.action_size, activation='softmax'))

        adam = Adam(lr=self.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam)
