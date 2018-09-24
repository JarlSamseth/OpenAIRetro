import os
import time
import traceback

import numpy as np

from models import INPUT_SHAPE, DQN_MASK

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
import gym
from collections import deque

MODEL_SAVE_DIR = "models"

def main():
    load_model = False
    sample_batch_size = 32
    episodes = 50000
    checkpoint = 1000
    replay_start_size = 50000

    env = gym.make('BreakoutDeterministic-v4')
    # env = gym.make('CartPole-v1')
    model_name = env.spec._env_name + ".h5"
    # env.mode="fast"
    # env = retro_contest.StochasticFrameSkip(env, n=4, stickprob=0.25)
    state_size = env.observation_space.shape
    action_size = env.action_space.n

    agent = DQN_MASK(state_size, action_size, INPUT_SHAPE)
    # agent = DQN_BOX(state_size, action_size, state_size)

    if (load_model):
        agent.load_model(MODEL_SAVE_DIR, model_name)

    # agent.load_model_from_file(BACKUP_FOLDER_NAME, WEIGHT_BACKUP_NAME)
    # agent.load_model(BACKUP_FOLDER_NAME, WEIGHT_BACKUP_NAME)

    stacked_frames = deque([np.zeros((INPUT_SHAPE[0:2]), dtype=np.int) for i in range(INPUT_SHAPE[-1])], maxlen=4)
    try:
        iteration = 0
        for episode in range(episodes):
            state = env.reset()

            done = False
            tot_reward = 0

            start = time.time()
            step = 0

            state, stacked_frames = agent.stack_frames(stacked_frames, state, True)

            while not done:
                action = agent.choose_best_action(state)
                next_state, reward, done, _ = env.step(action)

                env.render()

                # clipped_reward = np.clip(reward, -1., 1.)

                next_state, stacked_frames = agent.stack_frames(stacked_frames, next_state, False)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                tot_reward += reward

                step += 1
                iteration += 1
                agent.avg_q_max += np.amax(
                    agent.model.predict([state, np.ones((agent.batch_size, agent.action_size))]))

                if (iteration > replay_start_size):
                    agent.train_replay(sample_batch_size, iteration)
            end = time.time()

            if (iteration > replay_start_size):
                stats = [tot_reward, agent.avg_q_max / float(step), step,
                         agent.avg_loss / float(step), end - start]
                for i in range(len(stats)):
                    agent.sess.run(agent.update_ops[i], feed_dict={
                        agent.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = agent.sess.run(agent.summary_op)
                agent.summary_writer.add_summary(summary_str, episode + 1)

            print(
                "episode: %s score: %.2f memory length: %.0f epsilon: %.3f global_step:%.0f average_q:%.2f average loss:%.2f time: %.2f"
                % (episode, tot_reward, len(agent.memory), agent.exploration_rate, iteration,
                   agent.avg_q_max / float(step), agent.avg_loss / float(step), end - start))

            if (episode % checkpoint == 0 and episode is not 0):
                agent.save_model(MODEL_SAVE_DIR, model_name)

            agent.avg_q_max, agent.avg_loss = 0, 0

        agent.save_model(MODEL_SAVE_DIR, model_name)

    except Exception as e:
        print(e)
        traceback.print_exc()




if __name__ == '__main__':
    main()
