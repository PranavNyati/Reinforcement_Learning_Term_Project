# Group Number: 1
# Roll Number (Name of member): 20CS30037 (Pranav Nyati)
# Project Code: TCv2
# Project Title: Controlling a Toy Car around a Grid [Version 2]

import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import numpy as np
import pygame
import pandas as pd

from my_env_control import Environment_Control

class Q_learning:
    
    def __init__(self, env=None, gamma=0.95, e_greedy=0.05, alpha=0.05, init_type='zeros'):
        
        if env is None:
            raise ValueError('Environment not provided!')
        
        self.environment = env # the environment object for which the agent is learning
        self.gamma = gamma # discount factor
        self.epsilon = e_greedy # epsilon for epsilon-greedy action selection
        self.learning_rate = alpha # learning rate
        
        # Q(S,A) table is for storing the state-action values and E(S,A) table is for storing the eligibility traces
        self.q_table = {}
        self.e_table = {}
        self.initialize_q_e_values(init_type)
        
        self.final_optimal_policy = None
        self.final_optimal_value_function = None

    # initialize the Q(S,A) and E(S,A) values
    def initialize_q_e_values(self, init_type='zeros'):
        
        if init_type == 'zeros':
            for state in self.environment.state_value_func.keys():
                for action in self.environment.actions.keys():
                    self.q_table[(state, action)] = 0
                    self.e_table[(state, action)] = 0
                    
        elif init_type == 'random':
            for state in self.environment.state_value_func.keys():
                for action in self.environment.actions.keys():
                    self.q_table[(state, action)] = np.random.uniform(0, 1)
                    self.e_table[(state, action)] = 0
        
        else :
            raise ValueError("Invalid initialization type!")
                
        
    def select_next_action(self, state):
        # epsilon-greedy action selection based on the current state-action values
        
        if (state == self.environment.start_state): # only forward action is possible from the start state
            return 0  # forward action is represented by 0
        else:
            actions = np.array([self.q_table[(state, action)] for action in self.environment.actions.keys()])
            action_prob = np.array([self.epsilon/len(actions) for i in range(len(actions))])   # equal e/len(actions) probability of selecting each action (for exploration)
            action_prob[np.argmax(actions)] += 1 - self.epsilon # add (1-e) probability to the greedy action (for exploitation)
            return np.random.choice(len(actions), p=action_prob)
        
    
    def update_q_table(self, state, action, reward, next_state):
    
        # update the Q(S,A) table using the Q-learning update rule
        max_q_val = max([self.q_table[(next_state, next_action)] for next_action in self.environment.actions.keys()])
        self.q_table[(state, action)] += self.learning_rate * (reward + self.gamma * max_q_val - self.q_table[(state, action)])
        # self.q_table[(state, action)] += self.learning_rate * (reward + self.gamma * max([self.q_table[(next_state, next_action)] for next_action in self.environment.actions.keys()]) - self.q_table[(state, action)]) 
        
    def train_one_episode(self, episode_no):
        
        self.environment.reset() # reset the environment to the initial state
        state = self.environment.state
        done = False
        train_episode = [] # list to store the (state, action, reward) tuples for the current episode
        G = 0 # return for the current episode
        ctr = 0
        
        print('Episode {}:'.format(episode_no))
        
        # keep taking actions until the agent reaches the goal state
        while not done: 
            action = self.select_next_action(state)
            next_state, reward, done = self.environment.get_next_state_and_reward(state, action) # get the next state and reward based on the current state and action
            G += reward * (self.gamma ** ctr) # update the return for the current episode
            ctr += 1
            
            self.update_q_table(state, action, reward, next_state)
            train_episode.append((state, action, reward))
            state = next_state
            
        print('Episode {} completed as the agent reached the goal in {} steps'.format(episode_no, len(train_episode)))
        return train_episode, G
    
    def train_q_learning(self, num_episodes= 1000, eps=None):
        
        training_episodes = []
        episode_returns = []
        for episode_no in range(1, 1 + num_episodes):
            if eps is not None: # for decaying epsilon, to decrease the extent of exploration as the agent learns, and to exploit more
                self.epsilon = eps/(episode_no**0.3)
            
            episode, episode_return = self.train_one_episode(episode_no) # train one episode
            training_episodes.append(episode)
            episode_returns.append(episode_return)
            
        # compute the final optimal policy and value function
        self.final_optimal_policy = {}
        self.final_optimal_value_function = {}
        
        for state in self.environment.state_value_func.keys():
            actions = np.array([self.q_table[(state, action)] for action in self.environment.actions.keys()])
            self.final_optimal_policy[state] = np.argmax(actions) # the optimal action is the one with the maximum state-action value for the current state
            self.final_optimal_value_function[state] = np.max(actions) # the optimal value for the current state is the maximum state-action value for the current state
            
        return training_episodes, episode_returns
    
    def print_optimal_value_function(self, textfile = None):
        
        if self.final_optimal_value_function is None:
            print('Value function not computed yet')
            return
        
        if (self.environment is not None and textfile is None):
            for state in self.final_optimal_value_function.keys():
                print('State: %s, Optimal Value: %f' % (state, self.final_optimal_value_function[state]))
        
        elif (self.environment is not None and textfile is not None):
            f = open(textfile, 'w')
            for state in self.final_optimal_value_function.keys():
                f.write('State: %s, Optimal Value: %f' % (state, self.final_optimal_value_function[state]))
                f.write('\n')
                f.write('State: %s, Optimal Action: %s' % (state, self.final_optimal_policy[state]))
                f.write('\n\n')
            f.close()
        
        state_values = np.zeros((4, self.environment.grid_size, self.environment.grid_size), dtype=np.float32)

        #declare a 3D array to store the optimal actions
        optimal_actions = np.ones((4, self.environment.grid_size, self.environment.grid_size), dtype=np.int32)
        
        for state in self.final_optimal_value_function.keys():
            state_values[self.environment.directions.index(state[2]), state[1], state[0]] = self.final_optimal_value_function[state]
            optimal_actions[self.environment.directions.index(state[2]), state[1], state[0]] = self.final_optimal_policy[state]
            
        # Plot for the state values learned by the agent through Q-Learning
        for i in range(4):
            fig, axs = plt.subplots(1, 1, figsize=(20, 20))
            
            axs.imshow(state_values[i], interpolation='nearest', cmap='hot')
            axs.set_title('Q-Learning: State Values for Direction %s' % self.environment.directions[i])
            axs.set_xlabel('X')
            axs.set_ylabel('Y')
            axs.set_xticks(np.arange(self.environment.grid_size))
            axs.set_yticks(np.arange(self.environment.grid_size))
            axs.set_xticklabels(np.arange(1, self.environment.grid_size + 1))
            axs.set_yticklabels(np.arange(1, self.environment.grid_size + 1))
            for row in range(self.environment.grid_size):
                for col in range(self.environment.grid_size):
                    text = axs.text(col, row, round(state_values[i, row, col], 2), ha="center", va="center", color="blue", fontsize=15)
                    
            # plt.show()
            plt.savefig('Q_Learning_State_values_for_direction_%s.png' % self.environment.directions[i])
            plt.close()
            
        # Plot for the greedy actions learned by the agent through Q-Learning from each state
        for i in range(4):
            fig, axs = plt.subplots(1, 1, figsize=(20, 20))
            color_map = np.zeros((self.environment.grid_size, self.environment.grid_size), dtype=np.float32)
            
            axs.imshow(optimal_actions[i], interpolation='nearest')
            axs.set_title('Q_Learning: Optimal Actions for Direction %s' % self.environment.directions[i])
            axs.set_xlabel('X')
            axs.set_ylabel('Y')
            axs.set_xticks(np.arange(self.environment.grid_size))
            axs.set_yticks(np.arange(self.environment.grid_size))
            axs.set_xticklabels(np.arange(1, self.environment.grid_size + 1))
            axs.set_yticklabels(np.arange(1, self.environment.grid_size + 1))
            for row in range(self.environment.grid_size):
                for col in range(self.environment.grid_size):
                    arrow = ""
                    arrow = self.environment.directions_actions_mapping[(self.environment.directions[i], self.environment.actions[optimal_actions[i, row, col]])]
                    
                    text = axs.text(col, row, arrow, ha="center", va="center", color="black", fontsize=40)
                    
                    
            plt.savefig('Q_Learning_Optimal_actions_for_direction_%s.png' % self.environment.directions[i])
            plt.close()
            
    
    def plot_learning_curve(self, episode_returns, title = 'Performance of Q-Learning over episodes'):
        
        # Plot of the average return per episode as the agent learns (i.e. over the episodes)
        plt.figure(figsize=(10, 5))
        
        avg_episode_returns = []
        ctr = 1
        for i in range(len(episode_returns)):
            avg_episode_returns.append(np.mean(episode_returns[:ctr]))
            ctr += 1
        
        plt.plot(avg_episode_returns)
        plt.title(title)
        plt.xlabel("Episode number")
        plt.ylabel("Average Return per episode")
        plt.grid(True)
        # plt.show()
        plt.savefig(title + '_gamma_0.95_maze_16.png')
    
if __name__ == "__main__":
    
    np.random.seed(0)
    env = Environment_Control(grid_size=16) # initialize the environment
    q_learner = Q_learning(env, gamma=0.95, e_greedy=0.05, alpha=0.02, init_type='zeros') # initialize the Q-Learning agent
    Num_episodes = 500
    training_episodes, episode_returns = q_learner.train_q_learning(num_episodes=Num_episodes, eps=None) # train the agent for the specified number of episodes
    
    # store the training episodes and episode returns in a csv file (pandas dataframe)
    # df = pd.DataFrame(columns=['Episode_No', 'Episode_Returns', 'Episode_Length'])
    # for i in range(Num_episodes):
    #     df.loc[i] = [i+1, episode_returns[i], len(training_episodes[i])]
    # df.to_csv('Q_Learning_Episode_Returns_gamma_0.95_maze_16.csv')
    
    print("Training using Q-Learning Done!")
    q_learner.plot_learning_curve(episode_returns, title = 'Performance of Q-Learning over episodes')
    
    file_name = 'Q_l_forward_reward_goal_2x_reward_gamma_0.95_maze_16.txt'
    q_learner.print_optimal_value_function(textfile = file_name)
    
    # simulate the greedy policy learned by the agent
    pygame.init()
    q_learner.environment.policy_simulation(q_learner.environment.start_state, q_learner.final_optimal_policy, refresh_rate=0.1)
    print("Policy Simulation Done")
    pygame.quit()