from turtle import st
import numpy as np

class Agent:
    def __init__(self,lr,eps_start,eps_end,eps_dec,no_states,no_actions,gamma):
        self.lr = lr
        self.eps_start = eps_start 
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.no_states = no_states
        self.no_actions = no_actions
        self.gamma = gamma

        self.Q = {}
        self.init_Q()


    def init_Q(self):
        for state in range(self.no_states):
            for action in range(self.no_actions):
                self.Q[(state,action)] = 0.0
    
    def choose_action(self,state):
        
        if np.random.random() < self.eps_start:
            action = np.random.choice([i for i in range(self.no_actions)])
        else:
            
            for a in range(self.no_actions):
                actions = np.array([self.Q[(state,a)]])
            action = np.argmax(actions)
        return action

    def decrement_epsilon(self):
        if self.eps_start > self.eps_end:
            self.eps_start = self.eps_start * self.eps_dec
        else:
            self.eps_start = self.eps_end
    
    def learn(self,state,action,reward,state_):
        
        for a in range(self.no_actions):
            actions = np.array([self.Q[(state_,action)]])
        action_max = np.argmax(actions)

        self.Q[(state,action)] = self.Q[(state,action)] + self.lr*(reward + self.gamma*action_max  - self.Q[(state,action)])
        self.decrement_epsilon()

        