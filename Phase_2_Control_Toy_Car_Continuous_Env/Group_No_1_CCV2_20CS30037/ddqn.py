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


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1, hidden_dim2, seed=0):
        super(QNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Config:
    
    # define all the configurations and parameters here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate = 5e-4
    buffer_size = int(1e5)
    batch_size = 64
    discount_factor = 0.99
    update_target_freq = 5
    tau = 1e-3
    epsilon_start = 0.2
    epsilon_end = 0.0002
    epsilon_decay_rate = 1e-6
    hidden_dim1 = 64
    hidden_dim2 = 64
    max_t = 8000 # maximum number of timesteps per episode
    max_episodes = 300 # maximum number of episodes to train the agent
    save_logs_freq = 25 # save logs after every save_logs_freq episodes
    save_model_ckpts_freq = 25 # save model checkpoints after every save_model_ckpts_freq episodes
    train_stats_print_freq = 5 # print training statistics after every train_stats_print_freq episodes
    save_model_path = './ddqn_train_logs/model_checkpoints/'
    save_logs_path = './ddqn_train_logs/logs/'
    

class Experience_Replay_Memory:
    
    def __init__(self, capacity, random_seed=0):
        
        """ Initialize the Experience Replay Memory object."""
        
        self.capacity = capacity
        # Deque is a list-like container with fast appends and pops on either end => double-ended queue has O(1) insertion and deletion at both ends
        # More efficient than list for append and pop from the end of the container
        self.memory = deque(maxlen=capacity)
        self.seed = random.seed(random_seed)
        self.experiences = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done_flag"])
        
    def add_experience(self, state, action, reward, next_state, done_flag):
        
        """Add a new experience to memory."""
        
        if len(self.memory) >= self.capacity:
            self.memory.popleft()
        self.memory.append(self.experiences(state, action, reward, next_state, done_flag))
            
    
    def sample(self, batch_size):
        
        """Randomly sample a batch of experiences from memory of size batch_size."""
        
        experiences = random.sample(self.memory, k=batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(Config.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(Config.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(Config.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(Config.device)
        done_flags = torch.from_numpy(np.vstack([e.done_flag for e in experiences if e is not None]).astype(np.uint8)).float().to(Config.device)
        
        return (states,actions,rewards,next_states,done_flags)
    
    def __len__(self):
        return len(self.memory)




class DDQN_Agent:
    
    def __init__(self, env=None, hidden_dim1=64, hidden_dim2=64, seed=0):
        
        """Initialize an Agent object."""
        

        self.seed = random.seed(seed)
        if env is not None:
            self.env = env
        else:
            self.env = ContinuousCarRadarEnv()
            
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
         
        # Q-Network initialization
        self.q_network = QNetwork(self.state_dim, self.action_dim, hidden_dim1, hidden_dim2, seed).to(Config.device)
        self.target_network = QNetwork(self.state_dim, self.action_dim, hidden_dim1, hidden_dim2, seed).to(Config.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=Config.learning_rate)
        
        self.memory = Experience_Replay_Memory(Config.buffer_size, seed)         # Replay memory buffer initialization
        self.t_step = 0                                                          # Initialize time step (for updating every 
        self.epsilon = Config.epsilon_start                                      # Initialize epsilon for epsilon-greedy action selection
        self.tau = Config.tau                                                    # Initialize tau for soft update of target parameters
        print("Device: ", Config.device)
        print("DDQN Agent initialized successfully!")


    def update_target_network(self, update_type='full'):
        
        """ 
        Full update target network parameters, if update_type = 'full' => θ_target = θ_local
        Soft update target network parameters , if update_type = 'soft' => θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        
        if update_type == 'full':
            """Full update model parameters."""
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        elif update_type == 'soft':
            for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
            
    
    def select_action(self, state):
        
        """Epsilon-greedy action selection.
        State: numpy array of the current state observations (Shape: (state_dim,))"""
        
        # unsqueeze(0)? => to add a batch dimension of 1 at the beginning of the tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(Config.device)
       
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state_tensor)
        self.q_network.train()
        
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_dim))
        
    
    def update_epsilon(self):
        
        if self.epsilon > Config.epsilon_end:
            self.epsilon -= Config.epsilon_decay_rate
            
    def step(self, state, action, reward, next_state, done_flag):
        
        self.memory.add_experience(state, action, reward, next_state, done_flag)
        
        # Lerarn every update_target_freq time steps
        self.t_step = (self.t_step + 1) % Config.update_target_freq
        
        # if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
        loss = 1e3
        if len(self.memory) > Config.batch_size:
            experiences = self.memory.sample(Config.batch_size)
            loss = self.learn_from_experience(experiences, Config.discount_factor)
            # self.update_epsilon()
            
        return loss
        
    def learn_from_experience(self, experiences, discount_factor=Config.discount_factor):
        
        """Update value parameters using given batch of experience tuples."""
        
        states, actions, rewards, next_states, done_flags = experiences
        
        self.q_network.train()
        self.target_network.eval()
        
        q_estimates_current = self.q_network(states).gather(1, actions)
        
        ## DOUBLE DQN UPDATE (Sample max action from local network but get its Q-value estimate from target network)
        self.q_network.eval()
        
        with torch.no_grad():
            # q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
            max_action_q_network = self.q_network(next_states).detach().max(1)[1].unsqueeze(1).long()
            q_targets_next = self.target_network(next_states).gather(1, max_action_q_network)
            
        self.q_network.train()
        
        # Compute expected Q values
        expected_Q_values_current = rewards + (discount_factor * q_targets_next * (1 - done_flags))
        
        # Compute loss
        loss = F.mse_loss(q_estimates_current, expected_Q_values_current)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network after every update_target_freq time steps
        # self.update_target_network(update_type='soft')
        if self.t_step == 0:
            self.update_target_network(update_type='soft')
            
        # print("Loss: ", loss.item())
        return loss.item()
    
    
    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)
    
    def train_DQN_agent(self, max_episodes=Config.max_episodes, max_t=Config.max_t, stats_file=None):
        
        """Deep Q-Learning.
        
        Params
        ======
            max_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
        """
        
        if stats_file == None:
            stats_file = open('train_stats.txt', 'w')
                    
        total_reward_logs = [] # list containing scores from each episode
        loss_logs = []
        
        reward_window = deque(maxlen=100) # last 100 scores for plotting
        loss_window = deque(maxlen=20)
        
        # Make logs and model checkpoints directory
        if not os.path.exists(Config.save_logs_path):
            os.makedirs(Config.save_logs_path)
        if not os.path.exists(Config.save_model_path):
            os.makedirs(Config.save_model_path)
        
        print("STARTING TO TRAIN THE AGENT!")
        
        for episode_i in range(max_episodes):
            print("EPISODE: ", episode_i+1, file=stats_file)
            print("\n", file=stats_file)
            
            print("EPISODE: ", episode_i+1)
            print("\n")
            
            curr_state = self.env.reset()
            episode_reward = 0
            loss = 0
            
            for t in range(max_t):

                done_flag = False
                action = self.select_action(state=curr_state)
                next_state, reward, done_flag, _ = self.env.step(action)
                loss = self.step(curr_state, action, reward, next_state, done_flag)
                curr_state = next_state
                episode_reward += reward
                
                if (t+1) % 500 == 0:
                    print("Time-step: ", t+1, ", Current reward: ", reward, file=stats_file)
                    print("Time-step: ", t+1, ", Current reward: ", reward)
                
                self.env.render()
                self.update_epsilon()
                
                if done_flag:
                    print("GOAL REACHED SUCCESSFULLY!")
                    print("GOAL REACHED SUCCESSFULLY!", file=stats_file)
                    break
                
            reward_window.append(episode_reward)
            loss_window.append(loss)
            total_reward_logs.extend([[episode_i, episode_reward]])
            loss_logs.extend([[episode_i, loss]])
            
            print("\n", file=stats_file)
            print("\n")
            
            if (episode_i+1) % Config.save_logs_freq == 0:
                pd.DataFrame(total_reward_logs, columns=['Episode', 'Total Reward']).to_csv(Config.save_logs_path + 'total_reward_logs.csv', index=False)
                pd.DataFrame(loss_logs, columns=['Episode', 'Loss']).to_csv(Config.save_logs_path + 'loss_logs.csv', index=False)
            
            if (episode_i+1) % Config.save_model_ckpts_freq == 0:
                self.save_model(Config.save_model_path + 'model_checkpoint_' + str(episode_i+1) + '.pth')
            
            if (episode_i+1) % Config.train_stats_print_freq == 0:
                # print current episode loss and reward 
                print('\rEpisode {}\tLoss: {:.2f}\tTotal Reward: {:.2f}\tEpsilon: {:.6f}'.format(episode_i+1, loss, episode_reward, self.epsilon))
                print('\rEpisode {}\tLoss: {:.2f}\tTotal Reward: {:.2f}\tEpsilon: {:.6f}'.format(episode_i+1, loss, episode_reward, self.epsilon), file=stats_file)
                
            
            
            if ((episode_i+1) >= 100) and ((episode_i+1) % Config.train_stats_print_freq == 0):
                print('\n')
                print('\rEpisode {}\tAverage Score over last 100 episodes: {:.2f}'.format(episode_i+1, np.mean(reward_window)))
                print('\n')
                
                print('\n', file=stats_file)
                print('\rEpisode {}\tAverage Score over last 100 episodes: {:.2f}'.format(episode_i+1, np.mean(reward_window)), file=stats_file)
                print('\rEpisode {}\tAverage Loss over last 20 episodes: {:.2f}'.format(episode_i+1, np.mean(loss_window)), file=stats_file)
                print('\n', file=stats_file)
                
                
        print("TRAINING COMPLETE!")
        
        stats_file.close()
        
        self.save_model(Config.save_model_path + 'final_model.pth')
        
        return total_reward_logs, loss_logs
                
    def load_trained_network(self, path):
        
        """Load trained network parameters."""
        
        self.q_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))
        
        print("Trained network parameters loaded successfully!")
        
    def test_trained_agent(self, max_episodes=20, max_t=Config.max_t):
        
        """Test the trained agent."""
        
        total_reward_logs = []
        
        for episode in range(max_episodes):
            
            curr_state = self.env.reset()
            episode_reward = 0
            
            for t in range(max_t):
                
                action = self.select_action(state=curr_state)
                next_state, reward, done_flag, _ = self.env.step(action)
                self.env.render()
                curr_state = next_state
                episode_reward += reward
                
                if done_flag:
                    print("GOAL REACHED SUCCESSFULLY!")
                    break
                
            total_reward_logs.extend([[episode, episode_reward]])
            
            print("EPISODE: ", episode+1, ", TOTAL REWARD: ", episode_reward, ", GOAL REACHED: ", done_flag)

        total_reward_logs = np.array(total_reward_logs)
        
        print('\n')
        print('TEST SIMULATION COMPLETE!')
        print("AVERAGE TOTAL REWARD: ", np.mean(total_reward_logs[:,1]))
        

def plot_train_curves(total_reward_logs, loss_logs):
    
    """Plot training curves."""
    
    total_reward_logs = np.array(total_reward_logs)
    loss_logs = np.array(loss_logs)
    
    fig = plt.figure(figsize=(20, 10))

    # create two separate subplots for loss and total reward
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    # plot total reward curve
    ax1.plot(total_reward_logs[:,0], total_reward_logs[:,1], color='blue')
    ax1.set_xlabel('Episode Number')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Total Reward vs Episode')
    ax1.grid(True)
    
    # plot loss curve in log scale
    ax2.plot(loss_logs[:,0], np.log(loss_logs[:,1] + 1e-6), color='red')
    ax2.set_xlabel('Episode Number')
    ax2.set_ylabel('Loss (in Log scale)')
    ax2.set_title('Loss vs Episode')
    ax2.grid(True)
    
    plt.savefig('DDQN_learning_curves.png')
       
if __name__ == "__main__":
        
        
    # INITIALIZE THE AGENT AND TRAIN IT
    
    # pygame.init()
    # env = ContinuousCarRadarEnv()
    # agent = DDQN_Agent(env=env, hidden_dim1=Config.hidden_dim1, hidden_dim2=Config.hidden_dim2, seed=0)
    # total_reward_logs, loss_logs = agent.train_DQN_agent(max_episodes=Config.max_episodes, max_t=Config.max_t)
    # plot_train_curves(total_reward_logs, loss_logs)
    # pygame.quit()
    
    
    # TEST THE TRAINED AGENT
    
    pygame.init()
    env = ContinuousCarRadarEnv()
    agent = DDQN_Agent(env=env, hidden_dim1=Config.hidden_dim1, hidden_dim2=Config.hidden_dim2, seed=0)
    agent.load_trained_network(Config.save_model_path + 'final_model.pth')
    agent.test_trained_agent(max_episodes=5, max_t=Config.max_t)
    pygame.quit()
    
    
    # PLOT LEARNING CURVES FROM SAVED LOGS
    
    # total_reward_logs = np.array(pd.read_csv(Config.save_logs_path + 'total_reward_logs.csv'))
    # loss_logs = np.array(pd.read_csv(Config.save_logs_path + 'loss_logs.csv'))
    # plot_train_curves(total_reward_logs, loss_logs)
    
    
     
        
        