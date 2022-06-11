import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import torchvision
import gym
from utils import plot_learning_curve

class NeuralNetwork(nn.Module):
    def __init__(self,input_dim,n_actions) -> None:
        super(NeuralNetwork,self).__init__()

        self.fc1 = nn.Linear(*input_dim,128)
        self.fc2 = nn.Linear(128,n_actions)

        self.optimizer = optim.Adam(self.parameters(),lr=0.001)
        self.loss = nn.MSELoss()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    
    def forward(self,state):
        #x_np = torch.from_numpy(state)
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)
        return actions

class Agent():
    def __init__(self,lr,eps_start,eps_end,eps_dec,input_dim,n_actions,gamma):
        self.epsilon = eps_start
        self.lr = lr
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.gamma = gamma

        self.Q = NeuralNetwork(self.input_dim,self.n_actions) #Why cant I just put n_states instead of self.n_states

        self.action_space = [i for i in range(self.n_actions)]

    def choose_action(self,state): # epsilon greedy
        
        random_num = np.random.random()

        if random_num > self.epsilon: # take greedy action
            state = torch.tensor(state,dtype=torch.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = torch.argmax(actions).item()
        
        else:
            state = torch.tensor(state,dtype=torch.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = np.random.choice([i for i in range(self.n_actions)])
        
        return action
    
    def decrement_epsilon(self):
        if self.epsilon > self.eps_end and self.epsilon > 0:
            self.epsilon = self.epsilon * self.eps_dec
        else:
            self.epsilon = self.epsilon*self.eps_dec

    def learn(self,state,action,reward,state_):
        state_tensor = torch.tensor(state,dtype=torch.float).to(self.Q.device)
        action_tensor = torch.tensor(action).to(self.Q.device)
        reward_tensor = torch.tensor(reward).to(self.Q.device)
        state__tensor = torch.tensor(state_, dtype=torch.float).to(self.Q.device)

        #We need to have a predicted Q value, a target Q value and the max Q value
        q_pred = self.Q.forward(state_tensor)[action_tensor] 
        q_max = torch.argmax(self.Q.forward(state__tensor)).item() #action we could have taken
        q_target = reward_tensor + self.gamma*(q_max) # action we actually took

        loss = self.Q.loss(q_target,q_pred)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()

    

if __name__ == "__main__":

    env = gym.make("CartPole-v1")
    n_games = 10000
    scores = []
    eps_history = []
    agent = Agent(lr=0.001,eps_start=1.0,eps_end=0.1,eps_dec=0.999995,gamma=0.9,input_dim=env.observation_space.shape,n_actions=env.action_space.n)
    observation = env.reset()

    for episode in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            #env.render()
            action = agent.choose_action(observation)
            observation_,reward,done,info = env.step(action)
            score += reward
            agent.learn(observation,action,reward,observation_)
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            #print("episode ", episode, "score %.lf avg score %.lf epsilon %.2f" % score,avg_score,agent.epsilon)
            print('episode ', episode, 'score %.1f avg score %.1f epsilon %.2f' %(score, avg_score, agent.epsilon))
    filename = "cartpole_naive_dqn.png"
    x = [episode+1 for i in range(n_games)]
    plot_learning_curve(x, scores, eps_history, filename)