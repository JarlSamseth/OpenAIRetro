import argparse
import os
import time
import traceback

import gym
import tensorflow as tf

from TfSummary import TfSummary
from models import DQNMask
from enum import Enum

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

MODEL_SAVE_DIR = "models"
INPUT_SHAPE = (84, 84, 4)


class STATUS(Enum):
    OBSERVING = 1
    EXPLORING = 2
    TRAINING = 3


def train(args):
    env = gym.make('BreakoutDeterministic-v4')
    model_name = env.spec._env_name + ".h5"
    state_size = env.observation_space.shape
    action_size = env.action_space.n

    episodes = 50000
    checkpoint = 500
    replay_start_step = 200

    summary = TfSummary("breakout_dqn",
                        ["Score/Episode", "Average_MaxQ/Episode", "Steps/Episode", "Average_Loss/Episode"])

    agent = DQNMask(state_size, action_size, INPUT_SHAPE, int(args.memory_size), replay_start_step, args.load_model)

    try:

        global_step = 0

        for episode in range(episodes):
            done = False
            dead = False

            score = 0
            step = 0
            start_life = 5

            start = time.time()

            env.reset()
            observation, _, _, _ = env.step(1)

            state, stacked_observations = agent.stack_observations(None, observation, is_new_episode=True)
            while not done:

                if args.render:
                    env.render()

                action = agent.act(state, global_step)
                observation, reward, done, info = env.step(action)

                # if start_life > info['ale.lives']:
                #     dead = True
                #     start_life = info['ale.lives']

                next_state, stacked_observations = agent.stack_observations(stacked_observations, observation,
                                                                            is_new_episode=False)
                agent.remember(state, action, reward, next_state, done)

                # If agent is dead, set the flag back to false, but keep the history unchanged,
                # to avoid to see the ball up in the sky
                # if dead:
                #     dead = False
                # else:
                #     state = next_state
                state = next_state

                score += reward
                step += 1
                global_step += 1

                agent.update_sum_qmax(state)

                if getStatus(agent, global_step, replay_start_step) != STATUS.OBSERVING:
                    agent.train_replay(global_step)

                if done:
                    end = time.time()

                    print(
                        "state: %s episode: %s score: %.2f memory length: %.0f/%.0f epsilon: %.3f global_step:%.0f "
                        "average_q:%.2f average loss:%.5f time: %.2f "
                        % (
                            getStatus(agent, global_step, replay_start_step).name, episode, score, len(agent.memory),
                            agent.memory.maxlen,
                            agent.exploration_rate, global_step,
                            agent.sum_q_max / float(step), agent.sum_loss / float(step), end - start))

                    if global_step > replay_start_step:
                        summary.add_to_summary({"Score/Episode": score,
                                                "Average_MaxQ/Episode": agent.sum_q_max / float(step),
                                                "Steps/Episode": step,
                                                "Average_Loss/Episode": agent.sum_loss / float(step)}, episode)

                    if episode % checkpoint == 0 and episode is not 0:
                        agent.save_model(MODEL_SAVE_DIR, model_name)

            agent.sum_q_max, agent.sum_loss = 0.0, 0.0

        agent.save_model(MODEL_SAVE_DIR, model_name)

    except Exception as e:
        print(e)
        traceback.print_exc()


def getStatus(agent, global_step, replay_start_step):
    if global_step <= replay_start_step:
        return STATUS.OBSERVING
    elif global_step <= (replay_start_step + agent.final_exploration_frame):
        return STATUS.EXPLORING
    else:
        return STATUS.TRAINING


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def test(args):
    print("Testing model: " + args.load_model)
    replay_start_step = 1

    env = gym.make('BreakoutDeterministic-v4')
    state_size = env.observation_space.shape
    action_size = env.action_space.n

    agent = DQNMask(state_size, action_size, INPUT_SHAPE, int(args.memory_size), replay_start_step, args.load_model)

    episode_number = 0
    global_step = 0

    # test how to deep copy a model
    '''
    model_copy = clone_model(model)    # only copy the structure, not the value of the weights
    model_copy.set_weights(model.get_weights())
    '''

    while episode_number < 5000:

        done = False
        dead = False
        # 1 episode = 5 lives
        score, start_life = 0, 5
        observe = env.reset()

        observe, _, _, _ = env.step(1)

        state, stacked_frames = agent.stack_observations(None, observe, True)
        while not done:
            env.render()
            time.sleep(0.01)
            # get action for the current history and go one step in environment
            action = agent.act(state, global_step)
            observe, reward, done, info = env.step(action)

            next_state, stacked_frames = agent.stack_observations(stacked_frames, observe, False)

            # if the agent missed ball, agent is dead --> episode is not over
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']
                print("life: " + str(start_life))

            score += reward
            # If agent is dead, set the flag back to false, but keep the history unchanged,
            # to avoid to see the ball up in the sky
            if dead:
                dead = False
            else:
                state = next_state

            # print("step: ", global_step)
            global_step += 1

            if done:
                episode_number += 1
                print('episode: {}, score: {}'.format(episode_number, score))


def main(argv):
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--load_model', default=False,
                        help='Set this to true if you want to load an saved model')
    parser.add_argument('--render', type=str2bool, nargs="?", const=True, default=False,
                        help='Render the game')
    parser.add_argument('--memory_size', default=50000,
                        help='Maximum memory size')
    parser.add_argument('--test', type=str2bool, nargs="?", const=True, default=False,
                        help='Only test a preloaded model')
    args = parser.parse_args()

    if args.test:
        test(args)
    else:
        train(args)


if __name__ == '__main__':
    tf.app.run()
