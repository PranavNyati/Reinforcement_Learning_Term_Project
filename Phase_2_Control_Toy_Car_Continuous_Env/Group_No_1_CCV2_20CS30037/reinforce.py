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

class PNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1, hidden_dim2):
        super(PNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, action_dim)
        
        # self.seed = random.seed(seed)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            
            # glorot initialization
            module.weight.data.normal_(mean=0.0, std=1.0)    
            nn.init.xavier_uniform_(module.weight.data)
            print("Weights init!")

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.softmax(x, dim=1)
        return x
    
class Config:
    
    # define all the configurations and parameters here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate = 1e-3
    discount_factor = 0.99
    hidden_dim1 = 128
    hidden_dim2 = 64
    max_t = 40000 # maximum number of timesteps per episode
    max_episodes = 160 # maximum number of episodes to train the agent
    save_logs_freq = 20 # save logs after every save_logs_freq episodes
    save_model_ckpts_freq = 25 # save model checkpoints after every save_model_ckpts_freq episodes
    train_stats_print_freq = 10 # print training statistics after every train_stats_print_freq episodes
    save_model_path = './reinforce_train_logs/model_checkpoints/'
    save_logs_path = './reinforce_train_logs/logs/'
    

class Reinforce_Agent:
    
    def __init__(self, env=None, hidden_dim1=64, hidden_dim2=64):
        """Init a REINFORCE agent object."""
        
        if env is not None:
            self.env = env
        else:
            self.env = ContinuousCarRadarEnv()
            
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        
        # Policy network initialization
        self.policy_network = PNetwork(self.state_dim, self.action_dim, hidden_dim1, hidden_dim2).to(Config.device)
        # self.policy_network.apply(self.policy_network._init_weights)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=Config.learning_rate)
        
        self.log_probs = []
        self.rewards = []
        self.done_list = []

        self.eps = np.finfo(np.float32).eps.item()

        print("Device: ", Config.device)
        print("Policy network initialized successfully!")
        
    
    def select_action(self, state, flag=0):
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(Config.device)
        action_probs = self.policy_network(state)
  
        # define a categorical distribution over the list of probabilities of actions
        dist = torch.distributions.Categorical(action_probs)
        # sample an action from the distribution
        action = dist.sample()
        if (flag==1):
            print("Action probs: ", action_probs)
            print("Action selected: ", action)
        # compute the log probability of the selected action in the distribution
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob
        
        
    def append_step_info(self, log_prob, reward, done_flag):
            
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.done_list.append(done_flag)    
    
    def compute_returns(self, final_value=0):
        
        G = final_value
        returns = []
        for step in reversed(range(len(self.rewards))):
            G = self.rewards[step] + Config.discount_factor * G * (1 - self.done_list[step])
            returns.insert(0, G) # insert the computed return at the beginning of the list
        
        return returns
    
    def reset_lists(self):
        
        self.log_probs = []
        self.rewards = []
        self.done_list = []
        
    def learn(self):
        
        policy_loss = []
        returns = self.compute_returns(final_value=0)
        episode_return = returns[0]
        
        returns = torch.tensor(returns)
        
    
        # normalize the returns
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        self.reset_lists()
        
        return policy_loss.item(), episode_return
    
    def save_model(self, path):
        torch.save(self.policy_network.state_dict(), path)
        
    def train_REINFORCE_agent(self, max_epsiodes=Config.max_episodes, max_t=Config.max_t, stats_file=None):
        
        if stats_file is None:
            stats_file = open("reinforce_train_stats.txt", "w")
            
        total_return_log = []
        loss_log = []
        
        return_window = deque(maxlen=20)
        loss_window = deque(maxlen=20)
        
        # if the logs directory does not exist, create it
        if not os.path.exists(Config.save_logs_path):
            os.makedirs(Config.save_logs_path)
        if not os.path.exists(Config.save_model_path):
            os.makedirs(Config.save_model_path)
            
        print("Starting training...")
        
        for episode_i in range(max_epsiodes):

            print("EPISODE: ", episode_i+1, file=stats_file)
            print("\n", file=stats_file)
            
            print("EPISODE: ", episode_i+1)
            # print("\n")
            
            state = self.env.reset()
            loss = 0.0
            self.reset_lists()
            num_steps = 0
            
            for t in range(max_t):
            # t=0
            # while True:
                
                flag=0
                t+=1
                if (t%(Config.max_t/10)==0):
                    print("Timestep: ", t)
                    flag=1
                
                state = (state - state.mean()) / (state.std() + self.eps)
                # print("Current state: ", state)
                action, log_prob = self.select_action(state, flag=flag)
                state, reward, done_flag, _ = self.env.step(action)
                
                num_steps += 1
                self.append_step_info(log_prob, reward, done_flag)
                self.env.render()
                
                if done_flag:
                    print("GOAL REACHED!")
                    print("GOAL REACHED!", file=stats_file)
                    break
            
            assert len(self.log_probs) == len(self.rewards) == len(self.done_list) == num_steps
            
            # train the agent after every episode
            loss, episode_return = self.learn()
            
            # append the episode reward and loss to the logs
            total_return_log.extend([[episode_i+1, episode_return]])
            loss_log.extend([[episode_i+1, loss]])
            return_window.append(episode_return)
            loss_window.append(loss)
                
            # print training statistics after every train_stats_print_freq episodes
            if (episode_i+1) % Config.train_stats_print_freq == 0:
                print("Episode: ", episode_i+1, " | Episode Length: ", num_steps, " | Episode Return: ", episode_return, " | Loss: ", loss)
                print("Episode: ", episode_i+1, " | Episode Length: ", num_steps, " | Episode Return: ", episode_return, " | Loss: ", loss, file=stats_file)
                print("\n")
                print("\n", file=stats_file)
                
                if (episode_i+1 >= Config.train_stats_print_freq):
                    print("Average Return (last ", Config.train_stats_print_freq, " episodes): ", np.mean(return_window))
                    print("Average Return (last ", Config.train_stats_print_freq, " episodes): ", np.mean(return_window), file=stats_file)
                    
                    print("Average Loss (last ", Config.train_stats_print_freq, " episodes): ", np.mean(loss_window))
                    print("Average Loss (last ", Config.train_stats_print_freq, " episodes): ", np.mean(loss_window), file=stats_file)
                    
                    print("\n")
                    print("\n", file=stats_file)
                    
            # save the logs after every save_logs_freq episodes
            if (episode_i+1) % Config.save_logs_freq == 0:
                pd.DataFrame(total_return_log, columns=['Episode', 'Total Return']).to_csv(Config.save_logs_path + 'total_return_log.csv', index=False)
                pd.DataFrame(loss_log, columns=['Episode', 'Loss']).to_csv(Config.save_logs_path + 'loss_log.csv', index=False)
                
                # save the plots of the logs till now to the logs directory
                plot_train_curves(total_return_log, loss_log, Config.save_logs_path + 'REINFORCE_learning_curves_ep_' + str(episode_i+1) + '.png')
                
                
            # save the model checkpoints after every save_model_ckpts_freq episodes
            if (episode_i+1) % Config.save_model_ckpts_freq == 0:
                self.save_model(Config.save_model_path + 'model_' + str(episode_i+1) + '.pth')
                
            
        print("Training completed!")
        stats_file.close()
        
        self.save_model(Config.save_model_path + 'final_model.pth')
        
        return total_return_log, loss_log
    
    def load_trained_network(self, path):
        
        """Load trained policy network parameters."""
        
        self.policy_network.load_state_dict(torch.load(path))        
        print("Trained network parameters loaded successfully!")
        
    def test_trained_agent(self, max_episodes=20, max_t=Config.max_t):
        
        total_return_log = []
        
        
        for episode in range(max_episodes):
            
        
            state = self.env.reset()
            self.reset_lists()
            total_reward = 0
            
            for t in range(max_t):
                
                # normalize the state vector
                state = (state - state.mean()) / (state.std() + self.eps)
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


def plot_train_curves(total_return_log, loss_log, savepath):
    
    """Plot training curves."""
    
    total_return_log = np.array(total_return_log)
    loss_log = np.array(loss_log)
    
    fig = plt.figure(figsize=(20, 10))

    # create two separate subplots for loss and total reward
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    # plot total reward curve
    ax1.plot(total_return_log[:,0], total_return_log[:,1], color='blue')
    ax1.set_xlabel('Episode Number')
    ax1.set_ylabel('Total Return')
    ax1.set_title('Total Return vs Episode')
    ax1.grid(True)
    
    # plot loss curve  
    ax2.plot(loss_log[:,0], loss_log[:,1], color='red')
    ax2.set_xlabel('Episode Number')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss vs Episode')
    ax2.grid(True)
    
    plt.savefig(savepath)
      
      
if __name__ == "__main__":
        
        
    # INITIALIZE THE AGENT AND TRAIN IT
    seed = 0
    random.seed(seed)
    
    # pygame.init()
    # env = ContinuousCarRadarEnv()
    # agent = Reinforce_Agent(env=env, hidden_dim1=Config.hidden_dim1, hidden_dim2=Config.hidden_dim2)
    # total_return_log, loss_log = agent.train_REINFORCE_agent(max_epsiodes=Config.max_episodes, max_t=Config.max_t)
    # plot_train_curves(total_return_log, loss_log, 'REINFORCE_learning_curves.png')
    # pygame.quit()
    
    
    # TEST THE TRAINED AGENT
    
    pygame.init()
    env = ContinuousCarRadarEnv()
    agent = Reinforce_Agent(env=env, hidden_dim1=Config.hidden_dim1, hidden_dim2=Config.hidden_dim2)
    agent.load_trained_network(Config.save_model_path + 'final_model.pth')
    agent.test_trained_agent(max_episodes=2, max_t=10000)
    pygame.quit()
    
    
    # PLOT LEARNING CURVES FROM SAVED LOGS
    
    # total_reward_logs = np.array(pd.read_csv(Config.save_logs_path + 'total_reward_logs.csv'))
    # loss_logs = np.array(pd.read_csv(Config.save_logs_path + 'loss_logs.csv'))
    # plot_train_curves(total_reward_logs, loss_logs)
                


            