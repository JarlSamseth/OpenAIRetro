from agent import DeepQAgent, DeepQLstmAgent
import gym
import numpy as np
from retro_contest.local import make
import datetime
import retro
import pandas as pd
MODEL_NAME="sonic_genesis_act1"
WEIGHT_BACKUP_NAME = MODEL_NAME+"_weights.h5"
BACKUP_FOLDER_NAME="sonic_genesis_act1_2018-05-23 19_27_13.029404"
class Trainer:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes = 1000
        self.env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
        obs = self.env.reset()
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        self.checkpoint = 100
        self.agent = DeepQLstmAgent(self.state_size, self.action_size)
        #self.agent.load_model_from_file(BACKUP_FOLDER_NAME, WEIGHT_BACKUP_NAME)
        self.backup_folder = MODEL_NAME+"_"+datetime.datetime.now().__str__().replace(":","_")
        #self.rewards=pd.DataFrame()

    def run(self):
        try:
            for index_episode in range(self.episodes):
                state = self.env.reset()
                state = np.reshape(state, (1,)+self.state_size)

                done = False
                tot_reward = 0

                while not done:
                    #self.env.render()
                    action = self.agent.act(state)
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, (1,)+self.state_size)
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    tot_reward += reward
                if (index_episode % self.checkpoint == 0 and index_episode is not 0):
                    self.agent.save_model_and_memory(WEIGHT_BACKUP_NAME, self.backup_folder)
                print("Episode {}# Score: {}".format(index_episode, tot_reward))
                self.agent.replay(self.sample_batch_size)
                self.env.reset()
        finally:
            self.agent.save_model_and_memory(WEIGHT_BACKUP_NAME, self.backup_folder)


class CartPole:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes = 1000
        self.env = gym.make('CartPole-v1')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.agent = DeepQAgent(self.state_size, self.action_size)
        self.agent.load_model_from_file(WEIGHT_BACKUP_NAME)

    def run(self):
        try:
            for index_episode in range(self.episodes):
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])
                done = False
                index = 0
                while not done:
                    self.env.render()
                    action = self.agent.act(state)
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index += 1
                #if (index % 100 == 0):
                print("Episode {}# Score: {}".format(index_episode, index + 1))
                self.agent.replay(self.sample_batch_size)
        finally:
            self.agent.save_model(WEIGHT_BACKUP_NAME)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
