import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os
import numpy as np
import time
import matplotlib.pyplot as plt


env = gym.make("FrozenLake-v1")

observation = env.reset()

no_of_games = 0

episodes = 1000
total_no_of_wins = 0

policy = {0: 1, 1: 1, 2: 2, 3:2, 4:2, 5:1, 6:2, 8:2, 9: 1, 10: 1, 13: 2, 14: 2}

for episode in range(1,episodes + 1):
    observation = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = policy[observation]
        observation,reward,done,info = env.step(action)
        score+= reward

        if score == 1:
            total_no_of_wins += 1
    
    print("Episode:{} Score: {}".format(episode,score))

print(str((total_no_of_wins/episodes)*100) + "%  win rate")


'''
for i in range(15):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    score += reward

    if done:
        no_of_games +=1
        observation = env.reset()


    print()
    '''
env.close()
