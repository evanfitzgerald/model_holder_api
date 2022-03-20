"""
Script containing the training and testing loop for DQNAgent
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

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from gym import spaces

class BasicEnv(gym.Env):
    def __init__(self):
        super(BasicEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
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

        # Apply action
        state_list = list(self.state)
        state_list[0] += action-1
        state_list[0] = max(0, min(state_list[0], 23))
        
        # Reduce movement limit by 1
        self.movement_limit -= 1
        
        # Calculate reward
        reward = -self.fit_y[state_list[0]] * (1.05 - random.randint(0, 1)/10)

        # Check if movement is done
        if self.movement_limit <= 0: 
            done = True
        else:
            done = False
        
        # Set placeholder for info
        info = {}
        self.state = tuple(state_list)
        # Return step information
        return self.state, reward, done, info
    
    def reset(self):
        '''
        self.state = (0, 0, 0, 0)   
        return self.state
        '''
        self.state = (random.randint(0, 23), random.randint(-1, 1), 0, 0)

        # Reset movement limit 
        self.movement_limit = 24

        return self.state


def fill_memory(env, dqn_agent, num_memory_fill_eps):
    """
    Function that performs a certain number of episodes of random interactions with the environment to populate the replay buffer

    Parameters
    ---
    env: gym.Env
        Instance of the environment used for training
    dqn_agent: DQNAgent
        Agent to be trained
    num_memory_fill_eps: int
        Number of episodes of interaction to be performed

    Returns
    ---
    none
    """

    for _ in range(num_memory_fill_eps):
        done = False
        state = env.reset()

        while not done:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            dqn_agent.memory.store(state=state, 
                                action=action, 
                                next_state=next_state, 
                                reward=reward, 
                                done=done)


def train(env, dqn_agent, num_train_eps, num_memory_fill_eps, update_frequency, batchsize, results_basepath, render=False):
    """
    Function to train the agent

    Parameters
    ---
    env: gym.Env
        Instance of the environment used for training
    dqn_agent: DQNAgent
        Agent to be trained
    num_train_eps: int
        Number of episodes of training to be performed
    num_memory_fill_eps: int
        Number of episodes of random interaction to be performed
    update_frequency: int
        Number of steps after which the target models must be updated
    batchsize: int
        Number of transitions to be sampled from the replay buffer to perform a learning step
    results_basepath: str
        Location where models and other result files are saved
    render: bool
        Whether to create a pop-up window display the interaction of the agent with the environment
    
    Returns
    ---
    none
    """

    fill_memory(env, dqn_agent, num_memory_fill_eps)
    # print('Memory filled. Current capacity: ', len(dqn_agent.memory))
    
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
            dqn_agent.memory.store(state=state, action=action, next_state=next_state, reward=reward, done=done)

            dqn_agent.learn(batchsize=batchsize)

            if step_cnt % update_frequency == 0:
                dqn_agent.update_target_net()

            state = next_state
            ep_score += reward
            step_cnt += 1

        dqn_agent.update_epsilon()

        reward_history.append(ep_score)
        current_avg_score = np.mean(reward_history[-100:]) # moving average of last 100 episodes

        # print('Ep: {}, Total Steps: {}, Ep: Score: {}, Avg score: {}; Epsilon: {}'.format(ep_cnt,                           step_cnt, ep_score, current_avg_score, epsilon_history[-1]))
        
        if current_avg_score >= best_score:
            dqn_agent.save_model('{}/dqn_model'.format(results_basepath))
            best_score = current_avg_score

    with open('{}/train_reward_history.pkl'.format(results_basepath), 'wb') as f:
        pickle.dump(reward_history, f)

    with open('{}/train_epsilon_history.pkl'.format(results_basepath), 'wb') as f:
        pickle.dump(epsilon_history, f)

def test_auto(kilowatts):
    json_obj = {}

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

    for day in range(0, 730):
        json_obj[day] = {'kw': 0, 'hours': [], 'cost': 0}
        hours_list = [i for i in range(24)]
        random.shuffle(hours_list)

        hour_count = random.randint(12, 18)

        for hour in hours_list:
            if hour_count <= 0:
                break
            if cost_rates[hour] != on_peak:
                hour_count -= 1
                value = random.randint(0, 10)
                json_obj[day]['hours'].append(hour)
                if value > 3:
                    json_obj[day]['kw'] += kilowatts[2]
                    json_obj[day]['cost'] += kilowatts[2] * cost_rates[hour]
                else:
                    json_obj[day]['kw'] += kilowatts[1]
                    json_obj[day]['cost'] += kilowatts[1] * cost_rates[hour]

        json_obj[day]['cost'] = round(json_obj[day]['cost'], 2)

    return json_obj

def test(env, dqn_agent, num_test_eps, seed, results_basepath, kilowatts, fit_y, hour_fit_alignment, optimal_hours, render=True):
    """
    Function to test the agent

    Parameters
    ---
    env: gym.Env
        Instance of the environment used for training
    dqn_agent: DQNAgent
        Agent to be trained
    num_test_eps: int
        Number of episodes of testing to be performed
    seed: int
        Value of the seed used for testing
    results_basepath: str
        Location where models and other result files are saved
    render: bool
        Whether to create a pop-up window display the interaction of the agent with the environment

    Returns
    ---
    none
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
    temp_info = []
    env.fit_y = fit_y

    json_obj = {}

    for ep in range(0, 730):
        json_obj[ep] = {'kw': 0, 'hours': [], 'cost': 0}
        
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
                        state_output = optimal_hours[i+1]
                    elif val > 0.1:
                        state_output = optimal_hours[i+2]
                    else:
                        state_output = optimal_hours[i+3]

            json_obj[ep]['kw'] += kilowatts
            json_obj[ep]['hours'].append(state_output)
            json_obj[ep]['cost'] += kilowatts * cost_rates[state_output]
            json_obj[ep]['cost'] = round(json_obj[ep]['cost'], 2)

        # print('Ep: {}, Score: {}, State: {}, Temp Arr: {}'.format(ep,score,state_output, temp_info))

    with open('{}/test_reward_history_{}.pkl'.format('dqn/results/DQN-vDRAFT_epsmax1.0_epsmin0.01_epsdec0.998_batchsize64_memcap10000_updfreq1000', seed), 'wb') as f:
        pickle.dump(reward_history, f)
    
    return json_obj
        
if __name__ ==  '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-train-eps', type=int, default=2000, help='specify the max episodes to train for (counts even the period of memory initialisation)')
    parser.add_argument('--num-test-eps', type=int, default=100, help='specify the max episodes to test for')
    parser.add_argument('--num-memory-fill-eps', type=int, default=20, help='number of timesteps after which learning should start (used to initialise the memory)')
    parser.add_argument('--update-frequency', type=int, default=1000, help='how frequently should the target network by updated')
    parser.add_argument('--train-seed', type=int, default=12321, help='seed to use while training the model')
    parser.add_argument('--test-seed', type=int, nargs='+', default=[456, 12, 985234, 123, 3202], help='seeds to use while testing the model')
    parser.add_argument('--discount', type=float, default=0.99, help='discounting value to determine how far-sighted the agent should be')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--eps-max', type=float, default=1.0, help='max value for epsilon')
    parser.add_argument('--eps-min', type=float, default=0.01, help='min value for epsilon')
    parser.add_argument('--eps-decay', type=float, default=0.998, help='amount by which to decay the epsilon value for annealing strategy')
    parser.add_argument('--batchsize', type=int, default=64, help='number of samples to draw from memory for learning')
    parser.add_argument('--memory-capacity', type=int, default=10000, help='define the capacity of the replay memory')
    parser.add_argument('--results-folder', type=str, help='folder where the models and results of the current run must by stored')
    parser.add_argument('--env-name', type=str, default='DQN-vDRAFT', help='environment in which to train the agent')
    parser.add_argument('--train', action='store_true', help='train the agent')
    parser.add_argument('--render', action='store_true', help='render the interaction')
    args = parser.parse_args()

    if args.train:

        os.environ['PYTHONHASHSEED']=str(args.train_seed)
        np.random.seed(args.train_seed)
        torch.manual_seed(args.train_seed)

        env = BasicEnv() # gym.make(args.env_name) # 
        env.seed(args.train_seed)
        env.action_space.np_random.seed(args.train_seed)

        if args.results_folder is None:
            args.results_folder = "results/{}_epsmax{}_epsmin{}_epsdec{}_batchsize{}_memcap{}_updfreq{}".format(args.env_name, args.eps_max, args.eps_min, args.eps_decay, args.batchsize, args.memory_capacity, args.update_frequency)

        os.makedirs(args.results_folder, exist_ok=True)

        dqn_agent = DQNAgent(device, 
                                env.observation_space.shape[0], 
                                env.action_space.n, 
                                discount=args.discount, 
                                eps_max=args.eps_max, 
                                eps_min=args.eps_min, 
                                eps_decay=args.eps_decay,
                                memory_capacity=args.memory_capacity,
                                lr=args.lr,
                                train_mode=True)

        train(env=env, 
                dqn_agent=dqn_agent, 
                results_basepath=args.results_folder, 
                num_train_eps=args.num_train_eps, 
                num_memory_fill_eps=args.num_memory_fill_eps, 
                update_frequency=args.update_frequency,
                batchsize=args.batchsize)

        env.close()

        dqn_agent.save_model('{}/dqn_model'.format(args.results_folder))
    
    else:
        for idx, seed in enumerate(args.test_seed):
            #print("Testing {}/{}, seed = {}".format(idx+1, len(args.test_seed), seed))
            os.environ['PYTHONHASHSEED']=str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            env = BasicEnv() # env = gym.make(args.env_name)
            env.seed(seed)
            env.action_space.np_random.seed(seed)

            dqn_agent = DQNAgent(device, 
                                env.observation_space.shape[0], 
                                env.action_space.n, 
                                discount=args.discount, 
                                eps_max=0.0, # epsilon values should be zero to ensure no exploration in testing mode
                                eps_min=0.0, 
                                eps_decay=0.0,
                                train_mode=False)

            dqn_agent.load_model('{}/dqn/dqn_model'.format(args.results_folder))

            test(env=env, dqn_agent=dqn_agent, num_test_eps=args.num_test_eps, seed=seed, results_basepath=args.results_folder, render=args.render)

            env.close()

def get_train_results(kilowatts, fit_y, hour_fit_alignment, optimal_hours):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-train-eps', type=int, default=2000, help='specify the max episodes to train for (counts even the period of memory initialisation)')
    parser.add_argument('--num-test-eps', type=int, default=100, help='specify the max episodes to test for')
    parser.add_argument('--num-memory-fill-eps', type=int, default=20, help='number of timesteps after which learning should start (used to initialise the memory)')
    parser.add_argument('--update-frequency', type=int, default=1000, help='how frequently should the target network by updated')
    parser.add_argument('--train-seed', type=int, default=12321, help='seed to use while training the model')
    parser.add_argument('--test-seed', type=int, nargs='+', default=[456, 12, 985234, 123, 3202], help='seeds to use while testing the model')
    parser.add_argument('--discount', type=float, default=0.99, help='discounting value to determine how far-sighted the agent should be')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--eps-max', type=float, default=1.0, help='max value for epsilon')
    parser.add_argument('--eps-min', type=float, default=0.01, help='min value for epsilon')
    parser.add_argument('--eps-decay', type=float, default=0.998, help='amount by which to decay the epsilon value for annealing strategy')
    parser.add_argument('--batchsize', type=int, default=64, help='number of samples to draw from memory for learning')
    parser.add_argument('--memory-capacity', type=int, default=10000, help='define the capacity of the replay memory')
    parser.add_argument('--results-folder', type=str, help='folder where the models and results of the current run must by stored')
    parser.add_argument('--env-name', type=str, default='DQN-vDRAFT', help='environment in which to train the agent')
    parser.add_argument('--train', action='store_true', help='train the agent')
    parser.add_argument('--render', action='store_true', help='render the interaction')

    args = parser.parse_args()

    for idx, seed in enumerate(args.test_seed):
        # print("Testing {}/{}, seed = {}".format(idx+1, len(args.test_seed), seed))
        os.environ['PYTHONHASHSEED']=str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = BasicEnv()
        env.seed(seed)
        env.action_space.np_random.seed(seed)

        dqn_agent = DQNAgent(device, 
                            env.observation_space.shape[0], 
                            env.action_space.n, 
                            discount=args.discount, 
                            eps_max=0.0, # epsilon values should be zero to ensure no exploration in testing mode
                            eps_min=0.0, 
                            eps_decay=0.0,
                            train_mode=False)

        dqn_agent.load_model('{}/dqn_model'.format('dqn/results/DQN-vDRAFT_epsmax1.0_epsmin0.01_epsdec0.998_batchsize64_memcap10000_updfreq1000'))

        results = test(env=env, dqn_agent=dqn_agent, num_test_eps=args.num_test_eps, seed=seed, results_basepath=args.results_folder, render=args.render, kilowatts=kilowatts, fit_y=fit_y, hour_fit_alignment=hour_fit_alignment, optimal_hours=optimal_hours)

        env.close()

        return results

def get_auto_train_results(kilowatts):

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-train-eps', type=int, default=2000, help='specify the max episodes to train for (counts even the period of memory initialisation)')
    parser.add_argument('--num-test-eps', type=int, default=100, help='specify the max episodes to test for')
    parser.add_argument('--num-memory-fill-eps', type=int, default=20, help='number of timesteps after which learning should start (used to initialise the memory)')
    parser.add_argument('--update-frequency', type=int, default=1000, help='how frequently should the target network by updated')
    parser.add_argument('--train-seed', type=int, default=12321, help='seed to use while training the model')
    parser.add_argument('--test-seed', type=int, nargs='+', default=[456, 12, 985234, 123, 3202], help='seeds to use while testing the model')
    parser.add_argument('--discount', type=float, default=0.99, help='discounting value to determine how far-sighted the agent should be')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--eps-max', type=float, default=1.0, help='max value for epsilon')
    parser.add_argument('--eps-min', type=float, default=0.01, help='min value for epsilon')
    parser.add_argument('--eps-decay', type=float, default=0.998, help='amount by which to decay the epsilon value for annealing strategy')
    parser.add_argument('--batchsize', type=int, default=64, help='number of samples to draw from memory for learning')
    parser.add_argument('--memory-capacity', type=int, default=10000, help='define the capacity of the replay memory')
    parser.add_argument('--results-folder', type=str, help='folder where the models and results of the current run must by stored')
    parser.add_argument('--env-name', type=str, default='DQN-vDRAFT', help='environment in which to train the agent')
    parser.add_argument('--train', action='store_true', help='train the agent')
    parser.add_argument('--render', action='store_true', help='render the interaction')

    args = parser.parse_args()

    for idx, seed in enumerate(args.test_seed):
        # print("Testing {}/{}, seed = {}".format(idx+1, len(args.test_seed), seed))
        os.environ['PYTHONHASHSEED']=str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = BasicEnv()
        env.seed(seed)
        env.action_space.np_random.seed(seed)

        dqn_agent = DQNAgent(device, 
                            env.observation_space.shape[0], 
                            env.action_space.n, 
                            discount=args.discount, 
                            eps_max=0.0, # epsilon values should be zero to ensure no exploration in testing mode
                            eps_min=0.0, 
                            eps_decay=0.0,
                            train_mode=False)

        dqn_agent.load_model('{}/dqn_model'.format('dqn/results/DQN-vDRAFT_epsmax1.0_epsmin0.01_epsdec0.998_batchsize64_memcap10000_updfreq1000'))

        results = test_auto(kilowatts)

        env.close()

        return results