import numpy as np
import matplotlib.pyplot as plt
import pygame
import random
import pandas as pd
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as torch_mp

from contcar_env import ContinuousCarRadarEnv
from collections import deque, namedtuple
from utils import shared_optim

os.environ["OMP_NUM_THREADS"] = "1"

class ACNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1, hidden_dim2, dropout=0.1):
        super(ACNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, action_dim)
        self.dropout = nn.Dropout(dropout)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
                
            nn.init.xavier_uniform_(module.weight.data)
            print("Weights init!")
            
            module.bias.data.fill_(0.0)

    def forward(self, state):
        x = self.fc1(state)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    

class Config:
    
    # define all the configurations and parameters here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate = 5e-4
    discount_factor = 0.99
    trace_decay = 0.95
    hidden_dim1 = 64
    hidden_dim2 = 64
    dropout = 0.1
    
    update_global_freq = 5 # update global network after every update_global_freq episodes
    
    max_t = 40000 # maximum number of timesteps per episode
    max_episodes = 300 # maximum number of episodes to train the agent
    save_logs_freq = 20 # save logs after every save_logs_freq episodes
    save_model_ckpts_freq = 20 # save model checkpoints after every save_model_ckpts_freq episodes
    train_stats_print_freq = 10 # print training statistics after every train_stats_print_freq episodes
    save_model_path = './a3c_train_logs/model_checkpoints/'
    save_logs_path = './a3c_train_logs/logs/'
    
class A3C_agent:
    
    def __init__(self, env=None, hidden_dim1=64, hidden_dim2=64, dropout=0.1):
     
        if env is not None:
            self.env = env
        else:
            self.env = ContinuousCarRadarEnv()
            
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        
        self.actor = ACNetwork(self.state_dim, self.action_dim, hidden_dim1, hidden_dim2, dropout).to(Config.device)
        self.critic = ACNetwork(self.state_dim, 1, hidden_dim1, hidden_dim2, dropout).to(Config.device)
        
        self.actor.apply(self.actor._init_weights)
        self.critic.apply(self.critic._init_weights)
        
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.done_list = []
        self.eps = np.finfo(np.float32).eps.item
        
    def select_action(self, state, flag=0):
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(Config.device)
        action_probs = F.softmax(self.actor(state), dim=1)
        
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        
        if flag == 1:
            print("Action probs: ", action_probs)
            print("Action selected: ", action)
            
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob
    
    def compute_loss(self, state, action, value_t):
        
        self.actor.train()
        self.critic.train()
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(Config.device)
        action = torch.tensor([action]).to(Config.device)
        
        action_probs = F.softmax(self.actor(state), dim=1)
        dist = torch.distributions.Categorical(action_probs)
        
        log_prob_act = dist.log_prob(action)
        
        value_estimate = self.critic(state)
        t_d = value_t - value_estimate
        value_loss = t_d.pow(2)
        
        policy_loss = -log_prob_act * t_d.detach().squeeze()
        
        total_loss = (policy_loss + value_loss).mean()
        
        return total_loss, policy_loss, value_loss
            
    def compute_returns(self, final_value=0, normalize=True):
        
        G=final_value
        returns = []
        for i in reversed(range(len(self.rewards))):
            G = self.rewards[i] + Config.discount_factor*G*(1-self.done_list[i])
            returns.insert(0, G)
            
        returns = torch.tensor(returns).to(Config.device)
        
        if normalize:
            returns = (returns - returns.mean()) / (returns.std() + self.eps())
            
        return returns
        
    def reset_lists(self):
        
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.done_list = []        
        
        
                    
        
class Worker(torch_mp.Process):
    def __init__(self, global_net, optim, global_eps, global_eps_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.global_net = global_net
        self.optim = optim
        self.global_max_episodes = Config.max_episodes
        self.max_steps = Config.max_t
        self.global_eps_r = global_eps_r
        self.res_queue = res_queue
        self.env = ContinuousCarRadarEnv()
        self.agent = A3C_agent(self.env)
        self.local_net = ACNetwork(self.agent.state_dim, self.agent.action_dim, Config.hidden_dim1, Config.hidden_dim2, Config.dropout)
        
        
    def run_worker(self):
        
        total_return_log = []
        policy_loss_log = []
        value_loss_log = []
        
        
        if not os.path.exists(Config.save_logs_path):
            os.makedirs(Config.save_logs_path)
        if not os.path.exists(Config.save_model_path):
            os.makedirs(Config.save_model_path) 
        
        print("Starting training...")
        
        for episode_i in self.global_max_episodes:
            print("EPISODE: ", episode_i+1)
            
            state = self.env.reset()
            loss = 0.0
            self.reset_lists()
            num_steps = 0
            
            for t in range(self.max_steps):
                
                flag=0
                t+=1
                if (t%(Config.max_t/10)==0):
                    print("Timestep: ", t)
                    flag=1
                    
                state = (state - state.mean())/(state.std() + self.agent.eps())
                
                action, log_prob = self.agent.select_action(state, flag)
                