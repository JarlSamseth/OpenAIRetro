import datetime
import logging as log
import os
import pickle
import random
import time
from collections import deque

import numpy as np
import pandas as pd
from keras.layers import Conv2D, Flatten, Activation
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from retro_contest.local import make

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
import gym
import skimage as sk
from skimage import transform
from PIL import Image
log.basicConfig(level=log.INFO)

MODEL_NAME = "breakout"
WEIGHT_BACKUP_NAME = MODEL_NAME + ".h5"
BACKUP_FOLDER_NAME = "sonic_genesis_act1_weights_2018-06-09 15_08_24.280086"
INPUT_SHAPE = (84, 84)


class Trainer:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes = 10000
        # self.env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
        self.env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
        self.env = gym.make("Breakout-v4")
        # self.env = gym.make("CartPole-v0")
        print('connecting to remote environment')
        # self.env = grc.RemoteEnv('tmp/sock')
        print('starting episode')
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        self.checkpoint = 1000
        self.agent = DeepConvQAgent(self.state_size, self.action_size)
        # self.agent.load_model_from_file(BACKUP_FOLDER_NAME, WEIGHT_BACKUP_NAME)
        # self.agent.load_model(BACKUP_FOLDER_NAME, WEIGHT_BACKUP_NAME)
        self.backup_folder = MODEL_NAME + "_" + datetime.datetime.now().__str__().replace(":", "_")
        # self.rewards=pd.DataFrame()
        self.metrics_reward = []
        self.replay_start_size = 50000
    def run(self):
        try:
            n_frame = 0
            for index_episode in range(self.episodes):
                state = self.env.reset()

                done = False
                tot_reward = 0

                start = time.time()
                index = 0

                while not done:
                    action = self.agent.act(state)
                    next_state, reward, done, _ = self.env.step(action)

                    self.env.render()
                    clipped_reward = np.clip(reward, -1, 1)
                    self.agent.remember(state, action, clipped_reward, next_state, done)
                    state = next_state
                    tot_reward += reward
                    index += 1
                    n_frame += 1
                if (index_episode % self.checkpoint == 0 and index_episode is not 0):
                    self.agent.save_model_and_memory(WEIGHT_BACKUP_NAME, self.backup_folder)
                end = time.time()
                print("Episode %s reward:%s AvgScore:%.3f, Time: %.3fs estimated left: %.2fmin n_frame=%f qval=%f" % (
                index_episode, tot_reward, tot_reward / index, end - start,
                (self.episodes - index_episode) * (end - start) / 60, n_frame, self.agent.target_f["qvalues"].iloc[-1]))
                self.metrics_reward.append(tot_reward / index)
                if (n_frame > self.replay_start_size):
                    self.agent.replay(self.sample_batch_size)

                # self.env.reset()
        except Exception as e:
            print(e)
        finally:
            self.agent.save_model_and_memory(WEIGHT_BACKUP_NAME, self.backup_folder)
            # plot_metric_reward(self.metrics_reward)
            plot_metric(self.agent.target_f)
            plt.show()


class BaseAgent:

    def __init__(self, state_size, action_size):
        self.weight_backup = "cartpole_weight.h5"
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.0005
        self.gamma = 0.95
        self.exploration_rate = 1.0
        self.exploration_min = 0.1
        self.exploration_decay = 0.9995
        self.memory = deque(maxlen=100000)
        self.randBinList = lambda n: [random.randint(0, 1) for b in range(1, n + 1)]
        self.model = None
        self.metrics = pd.DataFrame(columns=['reward', 'exploration_rate'])
        self.metrics_reward = []
        self.metrics_exp_rate = []
        self.target_f = pd.DataFrame({"qvalues": [0]})

    def __build_model(self):
        return None

    def preprocess_old(self, img):
        x_t1 = sk.color.rgb2gray(img)
        x_t1 = sk.transform.resize(x_t1, (80, 80))
        x_t1 = sk.exposure.rescale_intensity(x_t1, out_range=(0, 255))
        return x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
        # preprocessed_img = self.to_grayscale(self.downsample(img))
        # return np.reshape(preprocessed_img, (1,) + preprocessed_img.shape)

    def preprocess(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        x_t1 = processed_observation.astype('uint8')  # saves storage in experience memory
        return x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
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
        # self.metrics_reward.append(reward)

    def save_metric(self, reward, exploration_rate):
        self.metrics = self.metrics.append(pd.DataFrame({"reward": [reward], 'exploration_rate': [exploration_rate]}))

    def plot_metric(self):
        self.metrics = pd.DataFrame({"reward": self.metrics_reward})
        self.metrics.to_pickle("metrics.pickle")
        plt.figure()
        plt.title('reward')
        avg = self.metrics.groupby(np.arange(len(self.metrics)) // 2).mean()
        plt.plot(range(avg['reward'].size), avg['reward'].values)
        plt.show()

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
        if (np.random.uniform(high=0.9) < self.exploration_rate):
            return np.random.random_integers(0, self.action_size -1)
        q_values = self.model.predict(prepocessed_state).astype("int8")
        # print("Predicted value = %s , eps=%s, rand=%s"%(np.argmax(q_values), self.exploration_rate, randvalue))
        return np.argmax(q_values)

    def replay(self, sample_batch_size):

        if len(self.memory) < sample_batch_size:
            return

        sample_batch = self.get_sample_batch(sample_batch_size)
        states = None
        targets = None
        for state, action, reward, next_state, done in sample_batch:

            if done:
                q_value = reward
            else:
                next_targets = self.predict(next_state)
                q_value = reward + self.gamma * np.amax(next_targets)

            self.target_f = self.target_f.append(pd.DataFrame({"qvalues": [np.amax(q_value)]}))
            current_targets = np.zeros((1, self.action_size))
            current_targets[0][action] = q_value

            if states is None:
                targets = current_targets
                states = state
            else:
                targets = np.append(targets, current_targets, axis=0)
                states = np.append(states, state, axis=0)

            # state = np.reshape(state, (1,) + self.state_size)

        self.train(states, targets)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

    def get_sample_batch(self, sample_batch_size):
        return random.sample(self.memory, sample_batch_size)


class DeepConvQAgent(BaseAgent):

    def __init__(self, state_size, action_size):
        BaseAgent.__init__(self, state_size, action_size)

        self.value_size = 1
        self.__build_model()

    def __build_model(self):
        input = Input(shape=(INPUT_SHAPE + (1,)))
        x = Conv2D(16, 8, 8, subsample=(4, 4), activation='relu')(input)
        x = Conv2D(32, 4, 4, subsample=(2, 2), activation='relu')(x)
        x = Flatten()(x)
        x = Dense(units=256, activation='relu')(x)
        output = Dense(output_dim=self.action_size, activation='linear')(x)
        self.model = Model(inputs=input, outputs=output)
        adam = Adam(lr=self.learning_rate)
        self.model.compile(optimizer=adam, loss='mse')


class DeepConvQAgent2(BaseAgent):

    def __init__(self, state_size, action_size):
        BaseAgent.__init__(self, state_size, action_size)

        self.value_size = 1
        self.__build_model()

    def __build_model(self):
        model = Sequential()

        model.add(Conv2D(32, (8, 8), strides=(4, 4), input_shape=(INPUT_SHAPE + (1,))))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size))
        model.add(Activation('linear'))
        model.compile(optimizer=Adam(lr=self.learning_rate), loss="mse")
        self.model = model


def plot_metric_reward(data):
    metrics = pd.DataFrame({"reward": data})
    metrics.to_pickle("metrics_tot_reward.pickle")
    plt.figure()
    plt.title('tot_reward')
    avg = metrics.groupby(np.arange(len(metrics)) // 2).mean()
    plt.plot(range(avg['reward'].size), avg['reward'].values)


def plot_metric(data):
    plt.figure()
    plt.title("qvalues")
    plt.plot(np.arange(0, data["qvalues"].size), data["qvalues"])


def plot_metric_from_file(path=None):
    metrics = pd.read_pickle("metrics.pickle")
    plt.figure()
    plt.title('reward')
    avg = metrics.groupby(np.arange(len(metrics)) // 2).mean()
    plt.plot(range(avg['reward'].size), avg['reward'].values)
    plt.show()

if __name__ == "__main__":
    #plot_metric_from_file()
        trainer = Trainer()
        trainer.run()
