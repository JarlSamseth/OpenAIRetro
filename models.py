from collections import deque

import numpy as np
from keras.layers import Conv2D, Flatten, Multiply
from keras.layers import Input, Dense, Lambda
from keras.models import Model, Sequential
from keras.optimizers import Adam

from dqn_agent import DQNAgent


class DQNMask(DQNAgent):

    def __init__(self, state_size, action_size, input_shape, memory_size, replay_start_step, load_model):
        DQNAgent.__init__(self, state_size, action_size, replay_start_step, memory_size)

        self.input_shape = input_shape
        if load_model is not False:
            self.load_model(load_model)
        else:
            self.__build_model()

    def __build_model(self):
        input = Input(shape=self.input_shape, name='frames')
        actions_input = Input((self.action_size,), name='mask')
        normalized = Lambda(lambda x: x / 255.0)(input)
        conv_1 = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(normalized)
        conv_2 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv_1)
        conv_flattened = Flatten()(conv_2)
        hidden = Dense(units=256, activation='relu')(conv_flattened)
        output = Dense(output_dim=self.action_size, activation='linear')(hidden)
        filtered_output = Multiply(name="Qvalue")([output, actions_input])
        self.model = Model(input=[input, actions_input], output=filtered_output)
        optimizer = Adam(lr=0.00025)
        self.model.compile(optimizer=optimizer, loss=self.huber_loss)
        self.target_model = self.clone_model()
        print(self.model.summary())
