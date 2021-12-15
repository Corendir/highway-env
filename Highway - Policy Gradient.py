import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
import highway_env
from matplotlib import pyplot as plt
from gym.spaces import Discrete, Box
import random
from keyboard import read_key

env = gym.make("highway-fast-v0")
observation_dimension = env.observation_space.shape[0]
# print("observation_dimension: ", observation_dimension)
n_acts = env.action_space.n

#############################################
####### BUILDING A NEURAL NETWORK ###########
##### REPRESENTING A STOCHASTIC POLICY ######
#############################################

# net_stochastic_policy is a neural network representing a stochastic policy:
# it takes as inputs observations and outputs logits for each action
net_stochastic_policy = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3),
        nn.Flatten(),
        nn.Linear(in_features=36, out_features=72),
        nn.Tanh(),
        nn.Linear(in_features=72, out_features=n_acts)
        )

# policy inputs an observation and computes a distribution on actions
def policy(observation):
    if len(observation.shape) == 2:
        observation = observation.unsqueeze(0).unsqueeze(0)
    else:
        observation = observation.unsqueeze(1)
    # print("\ndimension en entrée du réseau : ", observation.shape)
    logits = net_stochastic_policy(observation).squeeze(1)
    # print("\ndimension des logits : ", logits.shape)
    return Categorical(logits=logits)

# choose an action (outputs an int sampled from policy)
def choose_action(observation):
    # print("\ndimension en entrée de choose_action : ", observation.shape)
    observation = torch.as_tensor(observation, dtype=torch.float32)
    return policy(observation).sample().item()

# make loss function whose gradient, for the right data, is policy gradient
def compute_loss(batch_observations, batch_actions, batch_weights):
    batch_logprobability = policy(batch_observations).log_prob(batch_actions)
    return -(batch_logprobability * batch_weights).mean()

### Constants for training
learning_rate = 1e-2
epochs = 20 # 50
batch_size = 100 # 5000
##########################

# make optimizer
optimizer = Adam(net_stochastic_policy.parameters(), lr = learning_rate)

#############################################
######### VANILLA POLICY GRADIENT ###########
#############################################

def vanilla_policy_gradient():
    for i in range(epochs):
        batch_observations = [] 
        batch_actions = []      
        batch_weights = []      
        batch_returns = []      
        batch_lengths = []      

        observation = env.reset()
        # print("observation actual dimension: ", observation.shape)
        # print("observation: ", observation)
        done = False            
        rewards_in_episode = []            # list for rewards in the current episode

        # First step: collect experience by simulating the environment with current policy
        while True:
            # print("vroum")
            batch_observations.append(observation.copy())

            # act in the environment
            action = choose_action(observation)
            observation, reward, done, _ = env.step(action)

            # save action, reward
            batch_actions.append(action)
            rewards_in_episode.append(reward)

            if done:
                print("\ncrash")
                # if episode is over, record info about episode
                episode_return, episode_length = sum(rewards_in_episode), len(rewards_in_episode)
                batch_returns.append(episode_return)
                batch_lengths.append(episode_length)

                # the weight for each logprobability(action|observation)
                batch_weights += [episode_return] * episode_length

                # reset episode-specific variables
                observation, done, rewards_in_episode = env.reset(), False, []

                # end experience loop if we have enough of it
                if len(batch_observations) > batch_size:
                    break

        # Step second: update the policy
        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(torch.as_tensor(batch_observations, dtype=torch.float32),
                                  torch.as_tensor(batch_actions, dtype=torch.int32),
                                  torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()

        print('epoch: %3d \t loss: %.3f \t return: %.3f \t episode_length: %.3f'%
                (i, batch_loss, np.mean(batch_returns), np.mean(batch_lengths)))

vanilla_policy_gradient()

###### EVALUATION ############

def run_episode(env, render = False):
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        if render:
            env.render()
        action = choose_action(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

policy_scores = [run_episode(env) for _ in range(10)] #100
print("Average score of the policy: ", np.mean(policy_scores))

go = True
while go:
    for _ in range(4):
        run_episode(env, True)
    print("Press r to restart simulation, otherwise another key")
    if read_key() != "r":
        go = False

env.close()
