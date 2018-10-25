import os
import time
import traceback

import numpy as np
import tensorflow as tf
from models import INPUT_SHAPE, DQN_MASK

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
import gym
from collections import deque
from TfSummary import TfSummary

MODEL_SAVE_DIR = "models"
import argparse

import sys

def main(args):

    episodes = 50000
    checkpoint = 500
    replay_start_step = 50000

    env = gym.make('BreakoutDeterministic-v4')
    model_name = env.spec._env_name + ".h5"
    state_size = env.observation_space.shape
    action_size = env.action_space.n

    agent = DQN_MASK(state_size, action_size, INPUT_SHAPE, int(args.memory_size), replay_start_step)

    if (args.load_model is not False):
        agent.load_model(args.load_model)

    summary = TfSummary("breakout_dqn",
                        ["Total_Reward/Episode", "Average_Max_Q/Episode", "Steps/Episode", "Average_Loss/Episode",
                         "Avg_duration_seconds/Episode"])

    stacked_frames = deque([np.zeros((INPUT_SHAPE[0:2]), dtype=np.int) for i in range(INPUT_SHAPE[-1])], maxlen=4)
    try:
        global_step = 0
        for episode in range(episodes):

            state = env.reset()

            done = False
            tot_reward = 0

            start = time.time()
            step = 0

            state, stacked_frames = agent.stack_frames(stacked_frames, state, True)
            start_life = 5
            dead = False
            while not done:
                action = agent.act(state, global_step)
                next_state, reward, done, info = env.step(action)

                if (args.render):
                    env.render()

                if start_life > info['ale.lives']:
                    dead = True
                start_life = info['ale.lives']

                next_state, stacked_frames = agent.stack_frames(stacked_frames, next_state, False)
                agent.remember(state, action, reward, next_state, dead)
                # If agent is dead, set the flag back to false, but keep the history unchanged,
                # to avoid to see the ball up in the sky
                if dead:
                    dead = False
                else:
                    state = next_state

                tot_reward += reward
                step += 1
                global_step += 1
                agent.avg_q_max += np.amax(
                    agent.model.predict([state, np.ones((agent.batch_size, agent.action_size))]))

                if (global_step > replay_start_step):
                    agent.train_replay(global_step)
            end = time.time()

            if (global_step > replay_start_step):
                summary.add_to_summary({"Total_Reward/Episode": tot_reward,
                                        "Average_Max_Q/Episode": agent.avg_q_max / float(step),
                                        "Steps/Episode": step,
                                        "Average_Loss/Episode": agent.avg_loss / float(step),
                                        "Avg_duration_seconds/Episode": end - start}, episode)

            print(
                "state: %s episode: %s score: %.2f memory length: %.0f/%.0f epsilon: %.3f global_step:%.0f average_q:%.2f average loss:%.5f time: %.2f"
                % (getState(agent, global_step, replay_start_step), episode, tot_reward, len(agent.memory), agent.memory.maxlen, agent.exploration_rate, global_step,
                   agent.avg_q_max / float(step), agent.avg_loss / float(step), end - start))

            if (episode % checkpoint == 0 and episode is not 0):
                agent.save_model(MODEL_SAVE_DIR, model_name)

            agent.avg_q_max, agent.avg_loss = 0, 0

        agent.save_model(MODEL_SAVE_DIR, model_name)

    except Exception as e:
        print(e)
        traceback.print_exc()


def getState(agent, global_step, replay_start_step):
    if global_step <= replay_start_step:
        return "observing"
    elif global_step <= replay_start_step + agent.final_exploration_frame:
        return "exploring"
    else:
        return "training"


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--load_model', default=False,
                        help='Set this to true if you want to load an saved model')
    parser.add_argument('--render', type=str2bool, nargs="?", const=True, default=False,
                        help='Render the game')
    parser.add_argument('--memory_size', default=50000,
                        help='Maximum memory size')
    args=parser.parse_args()
    tf.app.run(argv=args)
