from collections import deque

import numpy as np
from keras.layers import Conv2D, Flatten
from keras.layers import Input, Dense, Lambda, merge
from keras.models import Model, Sequential
from keras.optimizers import Adam

from dqn_agent import DQN_AGENT

INPUT_SHAPE = (84, 84, 4)


class DQN(DQN_AGENT):

    def __init__(self, state_size, action_size, input_shape):
        DQN_AGENT.__init__(self, state_size, action_size)

        self.input_shape = input_shape
        self.__build_model()

    def __build_model(self):
        input = Input(shape=self.input_shape)
        normalized = Lambda(lambda x: x / 255.0)(input)
        x = Conv2D(16, 8, 8, subsample=(4, 4), activation='relu')(normalized)
        x = Conv2D(32, 4, 4, subsample=(2, 2), activation='relu')(x)
        x = Flatten()(x)
        x = Dense(units=256, activation='relu')(x)
        output = Dense(output_dim=self.action_size, activation='linear')(x)
        self.model = Model(inputs=input, outputs=output)
        adam = Adam()
        self.model.compile(optimizer=adam, loss=self.huber_loss)
        self.target_model = self.clone_model(self.model)
        print(self.model.summary())


class DQN_MASK(DQN_AGENT):

    def __init__(self, state_size, action_size, input_shape):
        DQN_AGENT.__init__(self, state_size, action_size)

        self.input_shape = input_shape
        self.__build_model()

    def __build_model(self):
        input = Input(shape=self.input_shape, name='frames')
        actions_input = Input((self.action_size,), name='mask')
        normalized = Lambda(lambda x: x / 255.0)(input)
        x = Conv2D(16, 8, 8, subsample=(4, 4), activation='relu')(normalized)
        x = Conv2D(32, 4, 4, subsample=(2, 2), activation='relu')(x)
        x = Flatten()(x)
        x = Dense(units=256, activation='relu')(x)
        output = Dense(output_dim=self.action_size, activation='linear')(x)
        filtered_output = merge([output, actions_input], mode='mul')
        self.model = Model(input=[input, actions_input], output=filtered_output)
        optimizer = Adam(lr=0.00025)
        self.model.compile(optimizer=optimizer, loss=self.huber_loss)
        self.target_model = self.clone_model(self.model)
        print(self.model.summary())


class DQN_BOX(DQN_AGENT):

    def __init__(self, state_size, action_size, input_shape):
        DQN_AGENT.__init__(self, state_size, action_size)

        self.input_shape = input_shape
        self.__build_model()

    def __build_model(self):
        model = Sequential()
        # 'Dense' is the basic form of a neural network layer
        # Input Layer of state size(4) and Hidden Layer with 24 nodes
        model.add(Dense(24, input_dim=self.input_shape[0], activation='relu'))
        # Hidden layer with 24 nodes
        model.add(Dense(24, activation='relu'))
        # Output Layer with # of actions: 2 nodes (left, right)
        model.add(Dense(self.action_size, activation='linear'))
        # Create the model based on the information above
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))

        print(model.summary())
        self.model = model
        self.target_model = model

    def stack_frames(self, stacked_frames, state, is_new_episode):
        # Preprocess frame

        if is_new_episode:
            # Clear our stacked_frames
            stacked_frames = deque([np.zeros((self.input_shape[0]), dtype=np.int) for i in range(4)],
                                   maxlen=4)

            # Because we're in a new episode, copy the same frame 4x
            stacked_frames.append(state)
            stacked_frames.append(state)
            stacked_frames.append(state)
            stacked_frames.append(state)

            # Stack the frames
            stacked_state = np.stack(stacked_frames, axis=1)

        else:
            # Append frame to deque, automatically removes the oldest frame
            stacked_frames.append(state)

            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(stacked_frames, axis=1)

        return stacked_state, stacked_frames
