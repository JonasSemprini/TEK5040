import numpy as np
import gym
import tensorflow as tf
import argparse
import time
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import sys

import matplotlib
matplotlib.use('QtAgg')

sys.path.append("../")


from car_race import baselines
from car_race.common import preprocess
from car_race.videofig.videofig import videofig


def show_episode(observations):

    def redraw_fn(i, axes):

        obs = observations[i]
        if not redraw_fn.initialized:
            redraw_fn.im = axes.imshow(obs, animated=True)
            redraw_fn.initialized = True
        else:
            redraw_fn.im.set_array(obs)

    redraw_fn.initialized = False

    videofig(len(observations), redraw_fn, play_fps=30)

def eval_policy(agent, num_steps, num_episodes, action_repeat=1, render_mode="rgb_array"):

    env = gym.make('CarRacing-v2', render_mode=render_mode)
    scores = []
    for i in range(num_episodes):
        # set seed so that we evaluate on same tracks each time
        #env.seed(100*i)
        observation,_ = env.reset(seed=100*i)
        rewards = []
        highres_observations = []
        t = 0
        # TODO: Render more nice image...
        while True:
            highres_observations.append(env.render())
            observation = preprocess(observation)
            action = agent(observation)
            action = np.array(action)[0]
            for _ in range(action_repeat):
                observation, reward, ter, trunc, info = env.step(action)
                rewards.append(reward)
                t += 1
                if ter or trunc:
                    break
                if num_steps >= 0 and t == num_steps:
                    break

            if ter or trunc:
                print("Episode finished after {} timesteps".format(t+1))
                break
            if num_steps >= 0 and t == num_steps:
                break

        score = sum(rewards)
        scores.append(score)
        if i == 0 or score > best_score:
            best_episode = highres_observations
            best_score = score

    env.close()

    return scores, best_episode

def main(agent, num_steps, num_episodes, action_repeat, render_mode, render_best=True):

    scores, best_episode = eval_policy(agent, num_steps, num_episodes, action_repeat=action_repeat, render_mode=render_mode)

    print("min, max : (%g, %g)"  % (np.min(scores), np.max(scores)))
    print("median, mean : (%g, %g)" % (np.median(scores), np.mean(scores)))

    if render_best:
        show_episode(best_episode)

def parse_args():
    """Parse command line argument."""

    parser = argparse.ArgumentParser("Show policy on 'Car-Race-V0' task.")
    parser.add_argument("--num_steps", type=int, default=200,
                        help="Maximum number of steps for episode.")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of steps to use for evaluation.")
    parser.add_argument("--action_repeat", type=int, default=1,
                        help="Number of steps to repeat each action.")
    parser.add_argument("--policy", default="random",
                        help="Either 'random' or 'straight' for baseline policies, or path to directory/file of saved model.")
    parser.add_argument("--render_mode", default="rgb_array",
                        help="Visualize (human) or not (rgb_array).")

    return parser.parse_args()

def get_agent(policy):

    if policy == "random":
        return baselines.random
    elif policy == "straight":
        return baselines.straight
    else:
        agent = tf.keras.models.load_model(policy)
        return agent

if __name__ == '__main__':

    args = parse_args()

    agent = get_agent(args.policy)
    main(agent, args.num_steps, args.num_episodes, args.action_repeat, args.render_mode)
