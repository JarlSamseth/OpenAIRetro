import datetime
import logging as log
import os
import pickle
import random
from collections import deque

import numpy as np
from keras.layers import Conv2D, Dense, Flatten, BatchNormalization, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from retro_contest.local import make
import gym_remote.exceptions as gre
import gym_remote.client as grc

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

log.basicConfig(level=log.INFO)

MODEL_NAME = "sonic_genesis_act1_weights"
WEIGHT_BACKUP_NAME = MODEL_NAME + ".h5"
BACKUP_FOLDER_NAME = "sonic_genesis_act1_weights_2018-06-09 15_08_24.280086"


class Trainer:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes = 1000
        #self.env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
        self.env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
        print('connecting to remote environment')
        # self.env = grc.RemoteEnv('tmp/sock')
        print('starting episode')
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        self.checkpoint = 100
        self.agent = DeepConvQAgent_SIMPLE(self.state_size, self.action_size)
        #self.agent.load_model_from_file(BACKUP_FOLDER_NAME, WEIGHT_BACKUP_NAME)
        # self.agent.load_model(BACKUP_FOLDER_NAME, WEIGHT_BACKUP_NAME)
        self.backup_folder = MODEL_NAME + "_" + datetime.datetime.now().__str__().replace(":", "_")
        # self.rewards=pd.DataFrame()

    def run(self):
        try:
            for index_episode in range(self.episodes):
                state = self.env.reset()

                done = False
                tot_reward = 0

                while not done:
                    action = self.agent.act(state)
                    next_state, reward, done, _ = self.env.step(action)

                    self.env.render()
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    tot_reward += reward
                if (index_episode % self.checkpoint == 0 and index_episode is not 0):
                    self.agent.save_model_and_memory(WEIGHT_BACKUP_NAME, self.backup_folder)
                print("Episode {}# Score: {}".format(index_episode, tot_reward))
                self.agent.replay(self.sample_batch_size)
                # self.env.reset()
        finally:
            self.agent.save_model_and_memory(WEIGHT_BACKUP_NAME, self.backup_folder)


class BaseAgent:

    def __init__(self, state_size, action_size):
        self.weight_backup = "cartpole_weight.h5"
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.exploration_rate = 1.0
        self.exploration_min = 0.1
        self.exploration_decay = 0.995
        self.memory = deque(maxlen=100000)
        self.randBinList = lambda n: [random.randint(0, 1) for b in range(1, n + 1)]
        self.model = None

    def __build_model(self):
        return None

    def preprocess(self, img):
        preprocessed_img = self.to_grayscale(self.downsample(img))
        return np.reshape(preprocessed_img, (1,) + preprocessed_img.shape)

    def to_grayscale(self, img):
        return np.mean(img, axis=-1, keepdims=1).astype(np.uint8)

    def downsample(self, img):
        return img[::2, ::2, ::]

    def load_model(self, backup_folder_name, model_name):
        log.info("Loading model and memory")
        if os.path.isfile(model_name):
            self.model.load_weights(model_name)
        else:
            log.warning("Could not find model on path")
        log.info("Loading successful")

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

    def train(self, state, target):
        self.model.train_on_batch(state, target)

    def predict(self, state):
        return self.model.predict(state)

    def calculate_target(self, reward, next_state):

        return reward + self.gamma * np.max(self.predict(next_state)[0])

    def remember(self, state, action, reward, next_state, done):
        prepocessed_state = self.preprocess(state)
        prepocessed_next_state = self.preprocess(next_state)
        self.memory.append((prepocessed_state, action, reward, prepocessed_next_state, done))

    def save_model_and_memory(self, model_name, dir_name):
        log.info("Saving model and memory")
        if (not os.path.isdir(script_dir + "\\" + dir_name)):
            os.mkdir(script_dir + "\\" + dir_name, 0o777)
        pickle.dump(self.memory, open(script_dir + "/" + dir_name + "/memory_file.pckl", "wb"))
        pickle.dump(self.exploration_rate, open(script_dir + "/" + dir_name + "/exploartion_rate.pckl", "wb"))
        self.model.save(script_dir + "/" + dir_name + "/" + model_name)
        log.info("Saving successful")

    def act(self, state):
        # Buttons: ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
        prepocessed_state = self.preprocess(state)
        if np.random.random() <= self.exploration_rate:
            return self.randBinList(self.action_size)
        q_values = self.model.predict(prepocessed_state).astype("int8")
        action = np.zeros((self.action_size,))
        action[np.argmax(q_values[0])] = 1
        return action

    def replay(self, sample_batch_size):

        if len(self.memory) < sample_batch_size:
            return

        sample_batch = self.get_sample_batch(sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:

            if done:
                Q_values = np.zeros((self.action_size, ))
                Q_values[np.argmax(action)] = reward
            else:
                Q_values = np.zeros((self.action_size, ))
                prediction = self.predict(next_state)
                Q_values[np.argmax(action)] = reward + self.gamma * np.max(prediction, axis=1)

            # state = np.reshape(state, (1,) + self.state_size)
            self.train(state, Q_values.reshape((1, self.action_size)))
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

    def get_sample_batch(self, sample_batch_size):
        return random.sample(self.memory, sample_batch_size)

class DeepConvQAgent_SIMPLE(BaseAgent):

    def __init__(self, state_size, action_size):
        BaseAgent.__init__(self, state_size, action_size)

        self.value_size = 1
        self.__build_model()

    def __build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, 8, 8, subsample=(4, 4),
                              input_shape=(int(self.state_size[0] / 2), int(self.state_size[1] / 2), 1)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, 4, 4, subsample=(2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, 4, 4, subsample=(1, 1)))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(units=512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(output_dim=self.action_size, activation='softmax'))

        adam = Adam(lr=self.learning_rate)
        self.model.compile(loss='mae', optimizer=adam)

class DeepConvQAgent(BaseAgent):

    def __init__(self, state_size, action_size):
        BaseAgent.__init__(self, state_size, action_size)

        self.value_size = 1
        self.__build_model()

    def __build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, 4, 4, subsample=(4, 4),
                              input_shape=(int(self.state_size[0] / 2), int(self.state_size[1] / 2), 1)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, 4, 4, subsample=(2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, 3, 3))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(units=512))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dense(output_dim=self.action_size))

        adam = Adam(lr=self.learning_rate)
        self.model.compile(loss='mae', optimizer=adam)


if __name__ == "__main__":
    try:
        trainer = Trainer()
        trainer.run()
    except Exception as e:
        print('exception thrown', e)
