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

from contcar_env import ContinuousCarRadarEnv
from collections import deque, namedtuple

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
    max_t = 15000 # maximum number of timesteps per episode
    max_episodes = 300 # maximum number of episodes to train the agent
    save_logs_freq = 20 # save logs after every save_logs_freq episodes
    save_model_ckpts_freq = 25 # save model checkpoints after every save_model_ckpts_freq episodes
    train_stats_print_freq = 20 # print training statistics after every train_stats_print_freq episodes
    save_model_path = './gae_train_logs/model_checkpoints/'
    save_logs_path = './gae_train_logs/logs/'

class GAE_Agent:
    
    def __init__(self, env=None, hidden_dim1=64, hidden_dim2=64, dropout=0.1):
        """Init a GAE agent object."""
        
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
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=Config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=Config.learning_rate)
        
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.done_list = []
        self.eps = np.finfo(np.float32).eps.item
        
        print("Device: ", Config.device)
        print("GAE Agent Initialized!")
        
    
    def select_action(self, state, flag=0):
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(Config.device)
        action_probs = F.softmax(self.actor(state), dim=1)
        
        # sample an action from the action distribution output by the actor network
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        
        if flag==1:
            print("Action probs: ", action_probs)
            print("Action selected: ", action)
            
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob
    
    def append_step_info(self, log_prob, value, reward, done):
        
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.done_list.append(done)
        
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
        
    def calculate_advantage(self, normalize=True):
        
        advantages = []
        curr_adv = 0
        next_val = 0
        
        # calculate advantages for each timestep of the episode
        for (value, reward, done) in zip(reversed(self.values), reversed(self.rewards), reversed(self.done_list)):
            done = 1 - done
            td_error = reward + Config.discount_factor*next_val*done - value.item()
            curr_adv = td_error + Config.discount_factor*Config.trace_decay*curr_adv*done
            next_val = value.item()
            advantages.insert(0, curr_adv)
            
        advantages = torch.tensor(advantages).to(Config.device)
        
        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps())
            
        return advantages
    
    def learn(self, advantages):
        
        advantages = advantages.detach()
        
        # compute returns for each timestep of the episode
        returns = self.compute_returns(normalize=False)
        episode_return = returns[0].item()
        returns = returns.detach()
        
        # compute policy loss and value loss
        policy_loss = - (advantages * self.log_probs).sum()
        value_loss = F.smooth_l1_loss(returns, self.values).sum()

        # take gradient step
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        policy_loss.backward()
        value_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        # reset the lists
        self.reset_lists()
        
        return policy_loss.item(), value_loss.item(), episode_return

    def save_model(self, path, model='actor'):
        
        if (model=='actor'):
            torch.save(self.actor.state_dict(), path)
            
        elif (model=='critic'):
            torch.save(self.critic.state_dict(), path)
        
    def train_GAE_agent(self, max_episodes=Config.max_episodes, max_t=Config.max_t, stats_file=None):
    
        if stats_file is None:
            stats_file = open("gae_train_stats.txt", "w")
            
        total_return_log = []
        policy_loss_log = []
        value_loss_log = []

        return_window = deque(maxlen=20)
        policy_loss_window = deque(maxlen=20)
        value_loss_window = deque(maxlen=20)
        
        # if the logs directory does not exist, create it
        if not os.path.exists(Config.save_logs_path):
            os.makedirs(Config.save_logs_path)
        if not os.path.exists(Config.save_model_path):
            os.makedirs(Config.save_model_path)
            
        print("Starting training...")
        
        for episode_i in range(max_episodes):

            print("EPISODE: ", episode_i+1, file=stats_file)
            print("\n", file=stats_file)
            
            print("EPISODE: ", episode_i+1)
            
            state = self.env.reset()
            loss = 0.0
            self.reset_lists()
            num_steps = 0

            for t in range(max_t):
                
                flag=0
                t+=1
                if (t%(Config.max_t/10)==0):
                    print("Timestep: ", t)
                    flag=1
            
                state = (state - state.mean()) / (state.std() + self.eps())
                action, log_prob = self.select_action(state, flag)
                value = self.critic(torch.from_numpy(state).float().unsqueeze(0).to(Config.device))
                
                state, reward, done_flag, _ = self.env.step(action)
                
                num_steps += 1
                self.append_step_info(log_prob, value, reward, done_flag)
                self.env.render()
                
                if done_flag:
                    print("GOAL REACHED!")
                    print("GOAL REACHED!", file=stats_file)
                    break
                
            assert len(self.log_probs)==len(self.values)==len(self.rewards)==len(self.done_list) == num_steps
            
            self.log_probs = torch.cat(self.log_probs)
            self.values = torch.cat(self.values).squeeze(-1)
            
            advantages = self.calculate_advantage()
            
            policy_loss, value_loss, episode_return = self.learn(advantages)
            
            total_return_log.extend([[episode_i+1, episode_return]])
            policy_loss_log.extend([[episode_i+1, policy_loss]])
            value_loss_log.extend([[episode_i+1, value_loss]])
            return_window.append(episode_return)
            policy_loss_window.append(policy_loss)
            value_loss_window.append(value_loss)
            
            # print training statistics after every train_stats_print_freq episodes
            if (episode_i+1) % Config.train_stats_print_freq == 0:
                print("Episode: ", episode_i+1, " | Episode Length: ", num_steps, " | Episode Return: ", episode_return, " | Policy Loss: ", policy_loss, " | Value Loss: ", value_loss)
                print("Episode: ", episode_i+1, " | Episode Length: ", num_steps, " | Episode Return: ", episode_return, " | Policy Loss: ", policy_loss, " | Value Loss: ", value_loss, file=stats_file)
                print("\n")
                print("\n", file=stats_file)
                
                if (episode_i+1 >= Config.train_stats_print_freq):
                    print("Average Return (last ", Config.train_stats_print_freq, " episodes): ", np.mean(return_window))
                    print("Average Return (last ", Config.train_stats_print_freq, " episodes): ", np.mean(return_window), file=stats_file)
                    
                    print("Average Policy Loss (last ", Config.train_stats_print_freq, " episodes): ", np.mean(policy_loss_window))
                    print("Average Policy Loss (last ", Config.train_stats_print_freq, " episodes): ", np.mean(policy_loss_window), file=stats_file)
                    
                    print("Average Value Loss (last ", Config.train_stats_print_freq, " episodes): ", np.mean(value_loss_window))
                    print("Average Value Loss (last ", Config.train_stats_print_freq, " episodes): ", np.mean(value_loss_window), file=stats_file)
                    
                    print("\n")
                    print("\n", file=stats_file)
                    
            # save logs after every save_logs_freq episodes
            if (episode_i+1) % Config.save_logs_freq == 0:
                pd.DataFrame(total_return_log, columns=['Episode', 'Total Return']).to_csv(Config.save_logs_path + 'total_return_log.csv', index=False)    
                pd.DataFrame(policy_loss_log, columns=['Episode', 'Policy Loss']).to_csv(Config.save_logs_path + 'policy_loss_log.csv', index=False)
                pd.DataFrame(value_loss_log, columns=['Episode', 'Value Loss']).to_csv(Config.save_logs_path + 'value_loss_log.csv', index=False)
                
                # save the plots of the logs till now to the logs directory
                plot_train_curves(total_return_log, policy_loss_log, value_loss_log, Config.save_logs_path + 'GAE_learning_curves_ep_' + str(episode_i+1) + '.png')
                
            # save model checkpoints after every save_model_ckpts_freq episodes
            if (episode_i+1) % Config.save_model_ckpts_freq == 0:
                self.save_model(Config.save_model_path + 'model_actor_ep_' + str(episode_i+1) + '.pth', model='actor')
                self.save_model(Config.save_model_path + 'model_critic_ep_' + str(episode_i+1) + '.pth', model='critic')
            
        print("Training complete!")
        stats_file.close()
        
        self.save_model(Config.save_model_path + 'final_model_actor.pth', model='actor')
        self.save_model(Config.save_model_path + 'final_model_critic.pth', model='critic')
        
        return total_return_log, policy_loss_log, value_loss_log
    
    def load_trained_network(self, path, model='actor'):
        
        """Load trained network parameters."""
        if (model=='actor'):
            self.actor.load_state_dict(torch.load(path))
            
        elif (model=='critic'):
            self.critic.load_state_dict(torch.load(path))
        
        
    def test_trained_agent(self, max_episodes=20, max_t=Config.max_t):
        
        total_return_log = []
        
        for episode in range(max_episodes):
            
            state = self.env.reset()
            self.reset_lists()
            total_reward = 0
            
            for t in range(max_t):
                
                # normalize the state vector
                # state = torch.from_numpy(state).float().unsqueeze(0).to(Config.device)
                state = (state - state.mean()) / (state.std() + self.eps())
                # print("Current state: ", state)
                action, _ = self.select_action(state)
                state, reward, done_flag, _ = self.env.step(action)
                self.rewards.append(reward)
                self.env.render()
                total_reward += reward
                
                if done_flag:
                    print("GOAL REACHED SUCCESSFULLY!")
                    break
                

            total_return_log.extend([[episode+1, total_reward]])
            
            print("Episode: ", episode+1, " | Episode Length: ", len(self.rewards), " | Episode Reward: ", total_reward)
            
        self.reset_lists()
        print('\n')
        print('Test Simulation Completed!')
        # print('Average Reward over ', max_episodes, ' episodes: ', np.mean(total_return_log[:, 1]))             
            
            
def plot_train_curves(total_return_log, policy_loss_log, value_loss_log, save_path):
    
    
    total_return_log = np.array(total_return_log)
    policy_loss_log = np.array(policy_loss_log)
    value_loss_log = np.array(value_loss_log)
    
    fig = plt.figure(figsize=(24, 16))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    # plot total return curve
    ax1.plot(total_return_log[:, 0], total_return_log[:, 1], color='blue')
    ax1.set_xlabel('Episode Number')
    ax1.set_ylabel('Total Return')
    ax1.set_title('Total Return vs Episode Number')
    ax1.grid(True)
    
    # plot policy loss curve
    ax2.plot(policy_loss_log[:, 0], policy_loss_log[:, 1], color='red')
    ax2.set_xlabel('Episode Number')
    ax2.set_ylabel('Policy Loss')
    ax2.set_title('Policy Loss vs Episode Number')
    ax2.grid(True)
    
    # plot value loss curve
    ax3.plot(value_loss_log[:, 0], value_loss_log[:, 1], color='green')
    ax3.set_xlabel('Episode Number')
    ax3.set_ylabel('Value Loss')
    ax3.set_title('Value Loss vs Episode Number')
    ax3.grid(True)
    
    plt.savefig(save_path)
    

            
if __name__ == "__main__":
        
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  
        
    # INITIALIZE THE AGENT AND TRAIN IT    
    # pygame.init()
    # env = ContinuousCarRadarEnv()
    # agent = GAE_Agent(env=env, hidden_dim1=Config.hidden_dim1, hidden_dim2=Config.hidden_dim2, dropout=Config.dropout)
    # total_return_log, policy_loss_log, value_loss_log = agent.train_GAE_agent(max_episodes=Config.max_episodes, max_t=Config.max_t)
    # plot_train_curves(total_return_log, policy_loss_log, value_loss_log, Config.save_logs_path + 'GAE_learning_curves.png')
    # pygame.quit()
    
    
    # TEST THE TRAINED AGENT
    pygame.init()
    env = ContinuousCarRadarEnv()
    agent = GAE_Agent(env=env, hidden_dim1=Config.hidden_dim1, hidden_dim2=Config.hidden_dim2, dropout=Config.dropout)
    agent.load_trained_network(Config.save_model_path + 'final_model_actor.pth', model='actor')
    agent.load_trained_network(Config.save_model_path + 'final_model_critic.pth', model='critic')
    agent.test_trained_agent(max_episodes=5, max_t=10000)
    pygame.quit()
    