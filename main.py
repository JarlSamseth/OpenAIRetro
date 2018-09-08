import datetime
import os
import time
import traceback

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from models import DQN, INPUT_SHAPE

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
import gym
from collections import deque

MODEL_NAME = "breakout"
WEIGHT_BACKUP_NAME = MODEL_NAME + ".h5"
BACKUP_FOLDER_NAME = "BreakoutDeterministic_2018-09-08 13_21_34.239414"


def main():
    sample_batch_size = 32
    episodes = 50000
    checkpoint = 1000
    replay_start_size = 50000

    env = gym.make('BreakoutDeterministic-v4')
    MODEL_NAME = env.spec._env_name
    # env.mode="fast"
    # env = retro_contest.StochasticFrameSkip(env, n=4, stickprob=0.25)
    state_size = env.observation_space.shape
    action_size = env.action_space.n

    agent = DQN(state_size, action_size, INPUT_SHAPE)
    # agent.load_model_from_file(BACKUP_FOLDER_NAME, WEIGHT_BACKUP_NAME)
    # agent.load_model(BACKUP_FOLDER_NAME, WEIGHT_BACKUP_NAME)
    backup_folder = MODEL_NAME + "_" + datetime.datetime.now().__str__().replace(":", "_")

    reward_merics = pd.DataFrame({"tot_reward": [0]})
    stacked_frames = deque([np.zeros((INPUT_SHAPE[0:2]), dtype=np.int) for i in range(INPUT_SHAPE[-1])], maxlen=4)
    try:
        iteration = 0
        for index_episode in range(episodes):
            state = env.reset()

            done = False
            tot_reward = 0

            start = time.time()
            index = 0

            state, stacked_frames = agent.stack_frames(stacked_frames, state, True)

            while not done:
                action = agent.choose_best_action(state, iteration)
                next_state, reward, done, _ = env.step(action)

                # env.render()
                # clipped_reward = np.clip(reward, -1., 1.)
                next_state, stacked_frames = agent.stack_frames(stacked_frames, next_state, False)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                tot_reward += reward
                index += 1
                iteration += 1

            reward_merics = reward_merics.append(pd.DataFrame({'tot_reward': [tot_reward]}))

            if (iteration > replay_start_size):
                agent.replay(sample_batch_size, iteration)

            end = time.time()
            print(
                "Episode %s reward:%s AvgScore:%.3f, Time: %.3fs estimated left: %.2fmin iteration=%.0f qval=%f exp_rate=%.3f" % (
                    index_episode, tot_reward, tot_reward / index, end - start,
                    (episodes - index_episode) * (end - start) / 60, iteration,
                    agent.get_metrics()["qvalues"].iloc[-1], agent.exploration_rate))

            if (index_episode % checkpoint == 0 and index_episode is not 0):
                agent.save_model_and_memory(WEIGHT_BACKUP_NAME, backup_folder)

        agent.save_model_and_memory(WEIGHT_BACKUP_NAME, backup_folder)

    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        # plot_metric_reward(metrics_reward)
        plot_metric(agent.get_metrics())
        plot_metric(reward_merics)
        plt.show()


def plot_metric(data):
    for col in data.columns:
        plt.figure(col)
        plt.title(col)
        plt.plot(np.arange(0, data[col].size), data[col])



if __name__ == '__main__':
    main()
