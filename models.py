from keras.layers import Conv2D, Flatten, Multiply
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import he_normal
from dqn_agent import DQNAgent


class DQNMask(DQNAgent):

    def __init__(self, state_size, action_size, input_shape, memory_size, replay_start_step, load_model):
        DQNAgent.__init__(self, state_size, action_size, replay_start_step, memory_size)

        self.input_shape = input_shape
        self.initializer = he_normal()
        if load_model is not False:
            self.__build_model()
            self.load_model(load_model)
        else:
            self.__build_model()

    def __build_model(self):
        input = Input(shape=self.input_shape, name='frames')
        actions_input = Input((self.action_size,), name='mask')
        normalized = Lambda(lambda x: x / 255.0)(input)
        conv_1 = Conv2D(16, 8, 8, subsample=(4, 4), activation='relu', kernel_initializer=self.initializer)(normalized)
        conv_2 = Conv2D(32, 4, 4, subsample=(2, 2), activation='relu', kernel_initializer=self.initializer)(conv_1)
        conv_flattened = Flatten()(conv_2)
        hidden = Dense(units=256, activation='relu', kernel_initializer=self.initializer)(conv_flattened)
        output = Dense(output_dim=self.action_size, activation='linear', kernel_initializer=self.initializer)(hidden)
        filtered_output = Multiply(name="Qvalue")([output, actions_input])
        self.model = Model(input=[input, actions_input], output=filtered_output)
        optimizer = Adam(lr=0.00001)
        self.model.compile(optimizer=optimizer, loss=self.huber_loss)
        self.target_model = self.clone_model()
        print(self.model.summary())


class DQNMaskV2(DQNAgent):

    def __init__(self, state_size, action_size, input_shape, memory_size, replay_start_step, load_model):
        DQNAgent.__init__(self, state_size, action_size, replay_start_step, memory_size)

        self.input_shape = input_shape
        self.initializer = he_normal()
        if load_model is not False:
            self.load_model(load_model)
        else:
            self.__build_model()

    def __build_model(self):
        input = Input(shape=self.input_shape, name='frames')
        actions_input = Input((self.action_size,), name='mask')
        normalized = Lambda(lambda x: x / 255.0)(input)
        conv_1 = Conv2D(32, 8, 8, subsample=(4, 4), activation='relu', kernel_initializer=self.initializer)(normalized)
        conv_2 = Conv2D(64, 4, 4, subsample=(2, 2), activation='relu', kernel_initializer=self.initializer)(conv_1)
        conv_3 = Conv2D(64, 3, 3, subsample=(1, 1), activation='relu', kernel_initializer=self.initializer)(conv_2)
        conv_flattened = Flatten()(conv_3)
        hidden = Dense(units=512, activation='relu', kernel_initializer=self.initializer)(conv_flattened)
        output = Dense(output_dim=self.action_size, activation='linear', kernel_initializer=self.initializer)(hidden)
        filtered_output = Multiply(name="Qvalue")([output, actions_input])
        self.model = Model(input=[input, actions_input], output=filtered_output)
        optimizer = Adam(lr=0.00001)
        self.model.compile(optimizer=optimizer, loss=self.huber_loss)
        self.target_model = self.clone_model()
        print(self.model.summary())
