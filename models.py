from keras.layers import Conv2D, Flatten
from keras.layers import Input, Dense, Lambda
from keras.models import Model
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
        adam = Adam(lr=self.learning_rate)
        self.model.compile(optimizer=adam, loss=self.huber_loss)
        self.target_model = self.copy_model(self.model)
