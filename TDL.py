import numpy as np


class Agent():
    def __init__(self,lr,gamma,n_actions,n_states,eps_start,eps_end,eps_dec):
        self.lr = lr
        self.gamma = gamma 
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec

        self.Q = {} #dictionary of state action pairs, only initialises it here

        self.init_Q()

    def init_Q(self): #initialise Q_table
        for state in range(self.n_states):  #Goes through each state and within each state, multiple actions and sets all those values to 0
            for action in range(self.n_actions): 
                self.Q[(state,action)] = 0.0 #Stores all state action pairs in a dictionary

    def choose_action(self,state):
        if np.random.random() < self.epsilon:
            action = np.random.choice([i for i in range(self.n_actions)])
        else:
            for a in range(self.n_actions):            
                actions = np.array(self.Q[(state,a)])     #list of elements correspoding to action values for a given state by looking at relevant quantities in Q_table
            action = np.argmax(actions)                #Then find index of maximum action from that list
        return action

    
    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon * self.eps_dec
        else:
            self.eps_min

    
    def learn(self,state,action,reward,state_):
        
        for a in range(self.n_actions):
            actions = np.array([self.Q[(state_,a)]])

        a_max = np.argmax(actions)
        self.Q[(state,action)] += self.lr*(reward + self.gamma*self.Q[(state_,a_max)] - self.Q[(state,action)])
        self.decrement_epsilon()