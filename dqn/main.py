"""
main.py

Contains main methods for training and testing the DQN agent. 

Several functions included in this file was based on the following reference.
Reference: https://github.com/saashanair/rl-series/tree/master/dqn 
"""

import json
import os
import csv
import random
import gym
import argparse
import numpy as np
import pickle

from dqn.dqn_agent import DQNAgent
from gym import spaces

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeferrableLoadEnv(gym.Env):
    """
    Defines a class to represent an environment which a deferrable load can exist.

    Deferrable appliances include instruments such as a washer, dryer, and dishwasher.
    """

    def __init__(self):
        """
        Uses gym.spaces objects to define action and observation spaces, initializes state.
        """

        super(DeferrableLoadEnv, self).__init__()
        self.state = (random.randint(0, 23), random.randint(-1, 1), 0, 0)
        self.movement_limit = 24
        self.fit_y = []

        self.action_space = gym.spaces.Discrete(3)
        high = np.array(
            [
                12,
                np.finfo(np.float32).max,
                1,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.int)

    def step(self, action):
        """
        Applies action selected by the model to the environment.
        """

        # Apply action to determine the new time setting
        state_list = list(self.state)
        state_list[0] += action - 1
        state_list[0] = max(0, min(state_list[0], 23))

        self.movement_limit -= 1

        reward = -self.fit_y[state_list[0]] * (1.05 - random.randint(0, 1) / 10)

        if self.movement_limit <= 0:
            done = True
        else:
            done = False

        info = {}

        self.state = tuple(state_list)

        return self.state, reward, done, info

    def reset(self):
        """
        Reset the environment by re-initializing the state and movement limit.
        """

        self.state = (random.randint(0, 23), random.randint(-1, 1), 0, 0)

        self.movement_limit = 24

        return self.state


class ContinuousLoadEnv(gym.Env):
    """
    Defines a class to represent an environment which a continuous load can exist.

    Continuous appliances include instruments such as a HVAC system.
    """

    def __init__(self):
        """
        Uses gym.spaces objects to define action and observation spaces, initializes state.
        """

        super(ContinuousLoadEnv, self).__init__()
        self.state = (0, random.randint(0, 11), 0, 0)
        self.movement_limit = 24
        self.temperatures = list(range(15, 26))

        self.action_space = gym.spaces.Discrete(3)
        high = np.array(
            [
                12,
                np.finfo(np.float32).max,
                1,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.int)

    def step(self, action):
        """
        Applies action selected by the model to the environment.
        """

        # Apply action to determine the new time and temperature setting
        state_list = list(self.state)
        hour = state_list[0]
        state_list[0] += 1
        temperature = self.temperatures[state_list[1]]

        self.movement_limit -= 1

        reward = -10

        # To account to user home temperature variance, the actionable value is supported using guided-random data.
        temperature += random.randint(-1, 0) + action - 2

        # store the actionable model kw value with current hour
        info = (hour, 2)

        if temperature > 23:
            state_list[1] -= 1
        elif temperature < 17:
            state_list[1] += 1
        else:
            state_list[1] += action - 1
            if temperature in {18, 19, 20, 21, 22}:
                info = (hour, 0)
                reward = 0
            else:
                reward = -5
                info = (hour, 1)

        if self.movement_limit <= 0:
            done = True
        else:
            done = False

        self.state = tuple(state_list)

        return self.state, reward, done, info

    def reset(self):
        """
        Reset the environment by re-initializing the state and movement limit.
        """

        self.state = (0, random.randint(4, 7), 0, 0)

        self.movement_limit = 24

        return self.state


def fill_memory(env, dqn_agent, num_memory_fill_eps):
    """
    Populates replay buffer with random interactions in the environment.
    """

    for _ in range(num_memory_fill_eps):
        done = False
        state = env.reset()

        while not done:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            dqn_agent.memory.store(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                done=done,
            )


def train(
    env,
    dqn_agent,
    num_train_eps,
    num_memory_fill_eps,
    update_frequency,
    batchsize,
    results_basepath,
    render=False,
):
    """
    Trains the provide DQN agent based on the inputted hyperparamters.
    """

    fill_memory(env, dqn_agent, num_memory_fill_eps)

    reward_history = []
    epsilon_history = []

    step_cnt = 0
    best_score = -np.inf

    for ep_cnt in range(num_train_eps):
        epsilon_history.append(dqn_agent.epsilon)

        done = False
        state = env.reset()

        ep_score = 0

        while not done:
            if render:
                env.render()

            action = dqn_agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            dqn_agent.memory.store(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                done=done,
            )

            dqn_agent.learn(batchsize=batchsize)

            if step_cnt % update_frequency == 0:
                dqn_agent.update_target_net()

            state = next_state
            ep_score += reward
            step_cnt += 1

        dqn_agent.update_epsilon()

        reward_history.append(ep_score)
        current_avg_score = np.mean(
            reward_history[-100:]
        )  # moving average of last 100 episodes

        if current_avg_score >= best_score:
            dqn_agent.save_model("{}/dqn_model".format(results_basepath))
            best_score = current_avg_score

    with open("{}/train_reward_history.pkl".format(results_basepath), "wb") as f:
        pickle.dump(reward_history, f)

    with open("{}/train_epsilon_history.pkl".format(results_basepath), "wb") as f:
        pickle.dump(epsilon_history, f)


def testContinuousAppliance(env, dqn_agent, kilowatts, render=False):
    """
    Tests a continuous appliance, generates 2 years worth of optimal forecast data.
    """

    # peak kWh charge ranges
    off_peak = 0.082
    mid_peak = 0.113
    on_peak = 0.170

    cost_rates = {
        0: off_peak,
        1: off_peak,
        2: off_peak,
        3: off_peak,
        4: off_peak,
        5: off_peak,
        6: off_peak,
        7: on_peak,
        8: on_peak,
        9: on_peak,
        10: on_peak,
        11: mid_peak,
        12: mid_peak,
        13: mid_peak,
        14: mid_peak,
        15: mid_peak,
        16: mid_peak,
        17: on_peak,
        18: on_peak,
        19: off_peak,
        20: off_peak,
        21: off_peak,
        22: off_peak,
        23: off_peak,
    }

    step_cnt = 0
    ret = {}

    for ep in range(0, 730):
        """
        Each day will store the total kilwatts used by that appliance,
        which hours of the day it was used, and the total cost based on the cost rates previously defined.
        """
        ret[ep] = {"kw": 0, "hours": [], "cost": 0}

        temperature = []
        score = 0
        done = False
        state = env.reset()
        while not done:

            if render:
                env.render()

            action = dqn_agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            temperature.append(info)
            score += reward
            state = next_state
            step_cnt += 1
            kilowatt_rating = kilowatts[temperature[len(temperature) - 1][1]]
            cost_rate = cost_rates[temperature[len(temperature) - 1][0]]
            ret[ep]["kw"] += kilowatt_rating
            if kilowatt_rating != 0:
                ret[ep]["hours"].append(temperature[len(temperature) - 1][0])
                ret[ep]["cost"] += kilowatt_rating * cost_rate
                ret[ep]["cost"] = round(ret[ep]["cost"], 2)

    return ret


def testDeferrableAppliance(
    env,
    dqn_agent,
    kilowatts,
    fit_y,
    hour_fit_alignment,
    optimal_hours,
    render=True,
):
    """
    Tests a deferrable appliance, generates 2 years worth of optimal forecast data.
    """

    # peak kWh charge ranges
    off_peak = 0.082
    mid_peak = 0.113
    on_peak = 0.170

    cost_rates = {
        0: off_peak,
        1: off_peak,
        2: off_peak,
        3: off_peak,
        4: off_peak,
        5: off_peak,
        6: off_peak,
        7: on_peak,
        8: on_peak,
        9: on_peak,
        10: on_peak,
        11: mid_peak,
        12: mid_peak,
        13: mid_peak,
        14: mid_peak,
        15: mid_peak,
        16: mid_peak,
        17: on_peak,
        18: on_peak,
        19: off_peak,
        20: off_peak,
        21: off_peak,
        22: off_peak,
        23: off_peak,
    }

    step_cnt = 0
    reward_history = []
    env.fit_y = fit_y

    ret = {}

    for ep in range(0, 730):
        ret[ep] = {"kw": 0, "hours": [], "cost": 0}

        hours_used = random.randint(1, 21)
        if hours_used >= 16:
            hours_used = 2
        elif hours_used <= 2:
            hours_used = 0
        else:
            hours_used = 1

        for h in range(hours_used):
            score = 0
            done = False
            state = env.reset()
            while not done:

                if render:
                    env.render()

                action = dqn_agent.select_action(state)
                next_state, reward, done, info = env.step(action)

                score += reward
                state = next_state
                step_cnt += 1

            reward_history.append(score)

            hour = hour_fit_alignment[state[0]]
            for i in range(len(optimal_hours)):
                if optimal_hours[i] == hour:
                    val = random.randint(0, 1)
                    if val > 0.25:
                        state_output = hour
                    elif val > 0.17:
                        state_output = optimal_hours[i + 1]
                    elif val > 0.1:
                        state_output = optimal_hours[i + 2]
                    else:
                        state_output = optimal_hours[i + 3]

            ret[ep]["kw"] += kilowatts[0]
            ret[ep]["hours"].append(state_output)
            ret[ep]["cost"] += kilowatts[0] * cost_rates[state_output]
            ret[ep]["cost"] = round(ret[ep]["cost"], 2)

    return ret


def generate_deferrable_appliance_data(
    kilowatts, fit_y, hour_fit_alignment, optimal_hours
):
    """
    Uses the deferrable load environment and model to generate optimal forecasted data.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-train-eps",
        type=int,
        default=2000,
        help="specify the max episodes to train for (counts even the period of memory initialisation)",
    )
    parser.add_argument(
        "--num-test-eps",
        type=int,
        default=100,
        help="specify the max episodes to test for",
    )
    parser.add_argument(
        "--num-memory-fill-eps",
        type=int,
        default=20,
        help="number of timesteps after which learning should start (used to initialise the memory)",
    )
    parser.add_argument(
        "--update-frequency",
        type=int,
        default=1000,
        help="how frequently should the target network by updated",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=12321,
        help="seed to use while training the model",
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        nargs="+",
        default=[456, 12, 985234, 123, 3202],
        help="seeds to use while testing the model",
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=0.99,
        help="discounting value to determine how far-sighted the agent should be",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--eps-max", type=float, default=1.0, help="max value for epsilon"
    )
    parser.add_argument(
        "--eps-min", type=float, default=0.01, help="min value for epsilon"
    )
    parser.add_argument(
        "--eps-decay",
        type=float,
        default=0.998,
        help="amount by which to decay the epsilon value for annealing strategy",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=64,
        help="number of samples to draw from memory for learning",
    )
    parser.add_argument(
        "--memory-capacity",
        type=int,
        default=10000,
        help="define the capacity of the replay memory",
    )
    parser.add_argument(
        "--results-folder",
        type=str,
        help="folder where the models and results of the current run must by stored",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="DQN-vDRAFT",
        help="environment in which to train the agent",
    )
    parser.add_argument("--train", action="store_true", help="train the agent")
    parser.add_argument("--render", action="store_true", help="render the interaction")

    args = parser.parse_args()

    for idx, seed in enumerate(args.test_seed):
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = DeferrableLoadEnv()
        env.seed(seed)
        env.action_space.np_random.seed(seed)

        dqn_agent = DQNAgent(
            device,
            env.observation_space.shape[0],
            env.action_space.n,
            discount=args.discount,
            eps_max=0.0,
            eps_min=0.0,
            eps_decay=0.0,
            train_mode=False,
        )

        dqn_agent.load_model(
            "{}/dqn_model".format(
                "dqn/results/DQN-vDRAFT_epsmax1.0_epsmin0.01_epsdec0.998_batchsize64_memcap10000_updfreq1000"
            )
        )
        results = testDeferrableAppliance(
            env=env,
            dqn_agent=dqn_agent,
            num_test_eps=args.num_test_eps,
            seed=seed,
            results_basepath=args.results_folder,
            render=args.render,
            kilowatts=kilowatts,
            fit_y=fit_y,
            hour_fit_alignment=hour_fit_alignment,
            optimal_hours=optimal_hours,
        )
        env.close()

        return results


def generate_continuous_appliance_data(kilowatts):
    """
    Uses the continuous load environment and model to generate optimal forecasted data.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-train-eps",
        type=int,
        default=2000,
        help="specify the max episodes to train for (counts even the period of memory initialisation)",
    )
    parser.add_argument(
        "--num-test-eps",
        type=int,
        default=100,
        help="specify the max episodes to test for",
    )
    parser.add_argument(
        "--num-memory-fill-eps",
        type=int,
        default=20,
        help="number of timesteps after which learning should start (used to initialise the memory)",
    )
    parser.add_argument(
        "--update-frequency",
        type=int,
        default=1000,
        help="how frequently should the target network by updated",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=12321,
        help="seed to use while training the model",
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        nargs="+",
        default=[456, 12, 985234, 123, 3202],
        help="seeds to use while testing the model",
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=0.99,
        help="discounting value to determine how far-sighted the agent should be",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--eps-max", type=float, default=1.0, help="max value for epsilon"
    )
    parser.add_argument(
        "--eps-min", type=float, default=0.01, help="min value for epsilon"
    )
    parser.add_argument(
        "--eps-decay",
        type=float,
        default=0.998,
        help="amount by which to decay the epsilon value for annealing strategy",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=64,
        help="number of samples to draw from memory for learning",
    )
    parser.add_argument(
        "--memory-capacity",
        type=int,
        default=10000,
        help="define the capacity of the replay memory",
    )
    parser.add_argument(
        "--results-folder",
        type=str,
        help="folder where the models and results of the current run must by stored",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="DQN-vDRAFT",
        help="environment in which to train the agent",
    )
    parser.add_argument("--train", action="store_true", help="train the agent")
    parser.add_argument("--render", action="store_true", help="render the interaction")

    args = parser.parse_args()

    for idx, seed in enumerate(args.test_seed):
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)

        env = ContinuousLoadEnv()
        env.seed(seed)
        env.action_space.np_random.seed(seed)

        dqn_agent = DQNAgent(
            device,
            env.observation_space.shape[0],
            env.action_space.n,
            discount=args.discount,
            eps_max=0.0,
            eps_min=0.0,
            eps_decay=0.0,
            train_mode=False,
        )

        dqn_agent.load_model(
            "{}/dqn_model".format(
                "dqn/results/DQN-vDRAFT_epsmax1.0_epsmin0.01_epsdec0.998_batchsize64_memcap10000_updfreq1000"
            )
        )
        results = testContinuousAppliance(
            env=env, dqn_agent=dqn_agent, render=args.render, kilowatts=kilowatts
        )
        env.close()

        return results


if __name__ == "__main__":
    """
    Used when the model is to be trained.

    Uses hyperparameters specified to train the model.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-train-eps",
        type=int,
        default=100000,
        help="specify the max episodes to train for (counts even the period of memory initialisation)",
    )
    parser.add_argument(
        "--num-test-eps",
        type=int,
        default=100,
        help="specify the max episodes to test for",
    )
    parser.add_argument(
        "--num-memory-fill-eps",
        type=int,
        default=20,
        help="number of timesteps after which learning should start (used to initialise the memory)",
    )
    parser.add_argument(
        "--update-frequency",
        type=int,
        default=1000,
        help="how frequently should the target network by updated",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=12321,
        help="seed to use while training the model",
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        nargs="+",
        default=[456, 12, 985234, 123, 3202],
        help="seeds to use while testing the model",
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=0.99,
        help="discounting value to determine how far-sighted the agent should be",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--eps-max", type=float, default=1.0, help="max value for epsilon"
    )
    parser.add_argument(
        "--eps-min", type=float, default=0.01, help="min value for epsilon"
    )
    parser.add_argument(
        "--eps-decay",
        type=float,
        default=0.998,
        help="amount by which to decay the epsilon value for annealing strategy",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=64,
        help="number of samples to draw from memory for learning",
    )
    parser.add_argument(
        "--memory-capacity",
        type=int,
        default=10000,
        help="define the capacity of the replay memory",
    )
    parser.add_argument(
        "--results-folder",
        type=str,
        help="folder where the models and results of the current run must by stored",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="DQN-vDRAFT",
        help="environment in which to train the agent",
    )
    parser.add_argument("--train", action="store_true", help="train the agent")
    parser.add_argument("--render", action="store_true", help="render the interaction")
    args = parser.parse_args()

    if args.train:

        os.environ["PYTHONHASHSEED"] = str(args.train_seed)
        np.random.seed(args.train_seed)
        torch.manual_seed(args.train_seed)

        env = DeferrableLoadEnv()
        env.seed(args.train_seed)
        env.action_space.np_random.seed(args.train_seed)

        if args.results_folder is None:
            args.results_folder = "results/{}_epsmax{}_epsmin{}_epsdec{}_batchsize{}_memcap{}_updfreq{}".format(
                args.env_name,
                args.eps_max,
                args.eps_min,
                args.eps_decay,
                args.batchsize,
                args.memory_capacity,
                args.update_frequency,
            )

        os.makedirs(args.results_folder, exist_ok=True)

        dqn_agent = DQNAgent(
            device,
            env.observation_space.shape[0],
            env.action_space.n,
            discount=args.discount,
            eps_max=args.eps_max,
            eps_min=args.eps_min,
            eps_decay=args.eps_decay,
            memory_capacity=args.memory_capacity,
            lr=args.lr,
            train_mode=True,
        )

        train(
            env=env,
            dqn_agent=dqn_agent,
            results_basepath=args.results_folder,
            num_train_eps=args.num_train_eps,
            num_memory_fill_eps=args.num_memory_fill_eps,
            update_frequency=args.update_frequency,
            batchsize=args.batchsize,
        )

        env.close()

        dqn_agent.save_model("{}/dqn_model".format(args.results_folder))
