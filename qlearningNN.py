from ast import main
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import gym
import numpy as np

class Network(nn.Module):
    def __init__(self,input_dims,no_of_actions) -> None:
        super(Network,self).__init__()

        self.fc1 = nn.Linear(*input_dims,128)
        self.fc2 = nn.Linear(128,no_of_actions)

        self.optimizer = optim.Adam(self.parameters(),lr=0.001)
        self.loss = nn.MSELoss()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device) # sets entire network to device

    def forward(self,state):
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)

        return actions

class Agent():
    def __init__(self, gamma, eps_start,eps_end,eps_dec, no_actions, no_states,lr): #no_states is number of input dimensions from the enviorment

        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.no_actions = no_actions
        self.no_states = no_states
        self.lr = lr

        #self.action_space = [i for i in range(no_actions)]

        self.Q = Network(self.no_states,self.no_actions)

    def choose_action(self,state):
        random_num = np.random.random()
        if random_num > self.epsilon: #take greedy action
            state = torch.tensor(state,dtype=torch.float).to(self.Q.device) # makes sure that state is tensor
            actions = self.Q.forward(state)
            action = torch.argmax(actions).item() # if you dont put .item() you will get back tensor and not value, which is not good for Gym env
            
        else:
            state = torch.tensor(state,dtype=torch.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = torch.random(actions).item()
            #OR
            #action = np.random.choice(self.action_space) # Makes things easier but want to see if above code works
        return action

    def decrement_epsilon(self):
        if self.epsilon > self.eps_end:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.epsilon = self.eps_end


    def learn(self,state,action,reward,state_):

        self.Q.optimizer.zero_grad()
        state_tensor = torch.tensor(state,dtype=torch.float).to(self.Q.device)
        action_tensor = torch.tensor(action).to(self.Q.device)
        reward_tensor = torch.tensor(reward).to(self.Q.device)
        state__tensor = torch.tensor(state,dtype=torch.float).to(self.Q.device)

        q_pred = self.Q.forward(state_tensor)[action_tensor] #predicted values for the current state of envioronment
        q_next = self.Q.forward(state__tensor).max() # action we could have taken
        q_target = reward_tensor + self.gamma*(q_next) #action we took

        loss = self.Q.loss(q_target,q_pred).to(self.Q.device) 
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()

        for a in range(self.no_actions):
            actions = self.Q.forward(state)
        action_max = torch.argmax(actions).item()

        self.Q = self.Q + self.lr*(reward + self.gamma*action_max  - self.Q)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(lr=0.9 ,gamma = 0.9, eps_start = 1.0, eps_end = 0.1, eps_dec=0.995,no_actions=env.action_space,no_states=env.observation_space.shape)
    observation = env.reset()

    no_of_games = 0

    episodes = 1000
    total_no_wins = 0

    for episode in range(1,episodes + 1):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score+=reward
            agent.learn(observation,action,reward,observation_)
            observation = observation_
        
        print("Episode:{} Score:{}".format(episode,score))
