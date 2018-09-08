import logging as log
import os
import random
from collections import deque

import numpy as np
import pandas as pd
from PIL import Image

from image_processor import ImageProcessor
from memory import RingBuf

log.basicConfig(level=log.INFO)

MODEL_NAME = "breakout"
WEIGHT_BACKUP_NAME = MODEL_NAME + ".h5"
BACKUP_FOLDER_NAME = "sonic_genesis_act1_weights_2018-06-09 15_08_24.280086"
INPUT_SHAPE = (84, 84, 4)

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in


class DQN_AGENT:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.00025
        self.gamma = 0.99
        self.exploration_rate = 1.0
        self.exploration_min = 0.1
        self.exploration_decay = 0.995
        self.final_exploration_frame = 1000000
        self.memory = RingBuf(1000000)  # deque(maxlen=200000)
        self.model = None
        self.n_stacked_frames = 4
        self.metrics = pd.DataFrame({"qvalues": [0]})
        self.image_processor = ImageProcessor()

    def __build_model(self):
        return None

    def preprocess(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE[0:2]).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE[0:2]
        x_t1 = processed_observation.astype('uint8')
        return x_t1

    def load_model_from_file(self, backup_folder_name, model_name):
        absolute_dir_path = script_dir + os.pathsep + backup_folder_name + os.pathsep + model_name
        log.info("Loading model and memory from dir=%s" % (absolute_dir_path))

        if os.path.isfile(absolute_dir_path + os.pathsep + model_name):
            self.model.load_weights(absolute_dir_path)
            self.exploration_rate = self.exploration_min
        else:
            log.warning("Could not find model on path")
        log.info("Loading successful")

    def save_model(self, model_name):
        self.model.save(model_name)

    def train(self, state, target):
        state = self.reshape_state(state)
        self.model.fit(state, target, epochs=1, verbose=0)

    def predict(self, state):
        state = self.reshape_state(state)
        return self.model.predict(state)

    def calculate_qvalue(self, reward, next_state):
        return reward + self.gamma * np.amax(self.predict(next_state))

    def remember(self, state, action, reward, next_state, done):
        # prepocessed_state = self.preprocess(state)
        self.memory.append((state, action, reward, next_state, done))

    def append_metrics(self, qvalue):
        self.metrics = self.metrics.append(pd.DataFrame({'qvalues': [qvalue]}))

    def get_metrics(self):
        return self.metrics

    def save_model_and_memory(self, model_name, dir_name):
        log.info("Saving model")
        if (not os.path.isdir(script_dir + "\\" + dir_name)):
            os.mkdir(script_dir + "\\" + dir_name, 0o777)
        self.model.save(script_dir + "/" + dir_name + "/" + model_name)
        log.info("Saving successful")

    def choose_best_action(self, state, iteration):
        # prepocessed_state = self.preprocess(state)
        if (np.random.uniform() < self.get_epsilon_for_iteration(iteration)):
            return np.random.random_integers(0, self.action_size - 1)
        state = self.reshape_state(state)
        q_values = self.model.predict(state).astype("int8")
        return np.argmax(q_values)

    def get_epsilon_for_iteration(self, iteration):
        if (iteration > self.final_exploration_frame):
            return 0
        return self.exploration_rate

    def replay(self, sample_batch_size):

        if len(self.memory) < sample_batch_size:
            return

        sample_batch = self.get_sample_batch(sample_batch_size)
        states = np.zeros((sample_batch_size,) + (INPUT_SHAPE))
        targets = np.zeros((sample_batch_size, self.action_size))
        for (state, action, reward, next_state, done), i in zip(sample_batch, range(sample_batch_size)):

            if done:
                q_value = reward
            else:
                next_targets = self.predict(next_state)
                q_value = reward + self.gamma * np.amax(next_targets)

            current_targets = self.predict(state)
            current_targets[0][action] = q_value
            self.train(state, current_targets)
            targets[i] = current_targets
            states[i] = state
            self.append_metrics(q_value)

        # self.train(states, targets)

        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

    def get_sample_batch(self, sample_batch_size):
        return random.sample(self.memory, sample_batch_size)

    def reshape_state(self, state):
        return np.reshape(state, (1,) + state.shape)

    def stack_frames(self, stacked_frames, state, is_new_episode):
        # Preprocess frame
        frame = self.preprocess(state)

        if is_new_episode:
            # Clear our stacked_frames
            stacked_frames = deque([np.zeros((INPUT_SHAPE[0:2]), dtype=np.int) for i in range(INPUT_SHAPE[-1])],
                                   maxlen=4)

            # Because we're in a new episode, copy the same frame 4x
            stacked_frames.append(frame)
            stacked_frames.append(frame)
            stacked_frames.append(frame)
            stacked_frames.append(frame)

            # Stack the frames
            stacked_state = np.stack(stacked_frames, axis=2)

        else:
            # Append frame to deque, automatically removes the oldest frame
            stacked_frames.append(frame)

            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(stacked_frames, axis=2)

        return stacked_state, stacked_frames
