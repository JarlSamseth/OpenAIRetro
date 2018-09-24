import logging as log
import os
import random
from abc import abstractmethod
from collections import deque

import numpy as np
import tensorflow as tf
from PIL import Image
from keras import backend as K
from keras.models import load_model
from keras.utils import to_categorical

log.basicConfig(level=log.INFO)

MODEL_NAME = "breakout"
WEIGHT_BACKUP_NAME = MODEL_NAME + ".h5"
BACKUP_FOLDER_NAME = "sonic_genesis_act1_weights_2018-06-09 15_08_24.280086"
INPUT_SHAPE = (84, 84, 4)

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in


class DQN_AGENT:

    def __init__(self, state_size, action_size):
        # environment settings
        self.state_size = state_size
        self.action_size = action_size
        self.input_shape = (84, 84, 4)
        self.batch_size = 32

        self.learning_rate = 0.001
        self.gamma = 0.99

        self.avg_q_max, self.avg_loss = 0, 0

        # parameters about epsilon
        self.exploration_rate_max, self.exploration_min = 1.0, 0.1
        self.exploration_rate = self.exploration_rate_max

        self.exploration_decay = 1. - 1 / 100000
        self.final_exploration_frame = 1000000
        self.memory = deque(maxlen=400000)
        self.n_stacked_frames = 4

        # model
        self.model = None
        self.target_model = None
        self.target_model_update_iteration = 1000

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/breakout_dqn', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    @abstractmethod
    def __build_model(self):
        return None

    def clone_model(self, model):
        """Returns a copy of a keras model."""
        model.save('tmp_model')
        return load_model('tmp_model', custom_objects={'huber_loss': self.huber_loss})

    def huber_loss(self, a, b, in_keras=True):
        error = a - b
        quadratic_term = error * error / 2
        linear_term = abs(error) - 1 / 2
        use_linear_term = (abs(error) > 1.0)
        if in_keras:
            # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
            use_linear_term = K.cast(use_linear_term, 'float32')
        return use_linear_term * linear_term + (1 - use_linear_term) * quadratic_term

    def preprocess(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE[0:2]).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE[0:2]
        x_t1 = processed_observation.astype('uint8')
        return self.reshape_state(x_t1)

    def load_model(self, directory, model_name):
        absolute_path = script_dir + os.sep + directory
        log.info("Loading model from directory=%s" % (absolute_path))
        model_path = absolute_path + os.sep + model_name
        if os.path.isfile(model_path):
            self.model.load_weights(model_path)
            self.exploration_rate = 0.1  # Have some exploration so it is not stuck on same place
            log.info("Loading successful")
        else:
            log.warning("Could not find model on path=" + model_path)

    def save_model(self, directory, model_name):
        log.info("Saving model")
        absolute_path = script_dir + os.sep + directory
        if (not os.path.isdir(absolute_path)):
            os.mkdir(absolute_path, 0o777)
        self.model.save(absolute_path + os.sep + model_name)
        log.info("Saving successful")

    def train(self, state, target):
        return self.model.fit(state, target, epochs=1, verbose=0)

    def predict(self, state):
        return self.model.predict(state)

    def remember(self, state, action, reward, next_state, done):
        # prepocessed_state = self.preprocess(state)
        self.memory.append((state, action, reward, next_state, done))

    def choose_best_action(self, state):
        if (np.random.uniform() < self.exploration_rate):
            return np.random.random_integers(0, self.action_size - 1)
        q_values = self.model.predict([state, np.ones((1, self.action_size))]).astype("int8")
        return np.argmax(q_values)

    def update_epsilon(self, iteration):
        a = (self.exploration_rate_max - self.exploration_min) / self.final_exploration_frame
        b = self.exploration_rate_max
        self.exploration_rate = np.max([b - a * iteration, self.exploration_min])

    def train_replay(self, sample_batch_size, iteration):

        if len(self.memory) < sample_batch_size:
            return

        self.update_epsilon(iteration)

        if (iteration % self.target_model_update_iteration is 0):
            self.target_model = self.clone_model(self.model)
            print("update")

        history = np.zeros(((self.batch_size,) + INPUT_SHAPE))
        next_history = np.zeros(((self.batch_size,) + INPUT_SHAPE))
        target = np.zeros((self.batch_size, self.action_size))
        action, reward, done = np.zeros((self.batch_size), dtype="uint8"), np.zeros((self.batch_size)), np.zeros(
            (self.batch_size))

        sample_batch = self.get_sample_batch()

        for i in range(self.batch_size):
            history[i] = sample_batch[i][0]
            next_history[i] = sample_batch[i][3]
            action[i] = sample_batch[i][1]
            reward[i] = sample_batch[i][2]
            done[i] = sample_batch[i][4]

        next_targets = self.target_model.predict([next_history, np.ones((self.batch_size, self.action_size))])

        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * np.amax(next_targets[i])

        result = self.train([history, to_categorical(action, num_classes=self.action_size)], target)
        loss = result.history["loss"][0]
        self.avg_loss += loss

    def get_sample_batch(self):
        return random.sample(self.memory, self.batch_size)

    def reshape_state(self, state):
        if (not list(state.shape).__len__() == list(self.input_shape).__len__()):
            return np.reshape(state, (1,) + state.shape)
        return state

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
            stacked_state = np.stack(stacked_frames, axis=3)

        else:
            # Append frame to deque, automatically removes the oldest frame
            stacked_frames.append(frame)

            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(stacked_frames, axis=3)

        return stacked_state, stacked_frames

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_step = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)
        episode_avg_duration = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Steps/Episode', episode_step)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)
        tf.summary.scalar('Avg duration/Episode in seconds', episode_avg_duration)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_step, episode_avg_loss, episode_avg_duration]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op
