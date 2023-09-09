import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import time
import numpy as np
import pygame

from my_env_control import Environment_Control


class SARSA_lambda:
    
    # Implementation of the SARSA(λ) algorithm for control 
    
    def __init__(self, env=None, gamma=0.95, e_greedy=0.05, alpha=0.05, lambda_wt_decay=0.2, init_type='zeros'):
        
        if env is None:
            raise ValueError("Environment not provided!")
        
        self.environment = env
        self.gamma = gamma  
        self.epsilon = e_greedy
        self.learning_rate = alpha
        self.lambda_decay = lambda_wt_decay
        
        # Q(S,A) table is for storing the state-action values and E(S,A) table is for storing the eligibility traces
        self.q_table = {}
        self.e_table = {}
        self.initialize_q_e_values(init_type)
        
        self.final_optimal_policy = None
        self.final_optimal_value_function = None
        
        
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
        
        if (state == self.environment.start_state):
            return 0 # only forward action is possible from the start state
        else:
            actions = np.array([self.q_table[(state, action)] for action in self.environment.actions.keys()])
            action_prob = np.array([self.epsilon/len(actions) for i in range(len(actions))])   # equal e/len(actions) probability of selecting each action (for exploration)
            action_prob[np.argmax(actions)] += 1 - self.epsilon # add (1-e) probability to the greedy action (for exploitation)
            return np.random.choice(len(actions), p=action_prob)
                
        
    def train_one_episode(self, episode_no):
        self.environment.reset()
        state = self.environment.state
        action = self.select_next_action(self.environment.state)
        done = False
        
        train_episode = []
        print('Episode {}:'.format(episode_no))
        while not done: 

            next_state, next_reward, done = self.environment.get_next_state_and_reward(state, action)
            next_action = self.select_next_action(next_state)
            
            delta = (next_reward + self.gamma * self.q_table[(next_state, next_action)]) - self.q_table[(state, action)]
            self.e_table[(state, action)] += 1
            for s in self.e_table.keys():
                self.q_table[s] += self.learning_rate * delta * self.e_table[s]
                self.e_table[s] *= self.gamma * self.lambda_decay
                
            train_episode.append((state, action, next_reward))
            state = next_state
            action = next_action
            
        print('Episode {} completed as the agent reached the goal in {} steps'.format(episode_no, len(train_episode)))
        
        return train_episode
    
    def train_sarsa_lambda(self, num_episodes= 1000, eps=None):
        
        training_episodes = []
        for episode_no in range(1, 1 + num_episodes):
            if eps is not None:
                self.epsilon = eps/(episode_no**0.3)
            
            episode = self.train_one_episode(episode_no)
            training_episodes.append(episode)
            
        # compute the final optimal policy and value function
        self.final_optimal_policy = {}
        self.final_optimal_value_function = {}
        
        for state in self.environment.state_value_func.keys():
            actions = np.array([self.q_table[(state, action)] for action in self.environment.actions.keys()])
            self.final_optimal_policy[state] = np.argmax(actions)
            self.final_optimal_value_function[state] = np.max(actions)
            
        return training_episodes
            
            
if __name__ == "__main__":
    
    env = Environment_Control(grid_size=20)
    sarsa_l = SARSA_lambda(env, gamma=0.90, e_greedy=0.05, alpha=0.02, lambda_wt_decay=0.2, init_type='zeros')
    training_episodes = sarsa_l.train_sarsa_lambda(num_episodes=2)
    print("Training using SARSA(λ) Done")
    # file_name = 'sarsa_l_forward_reward_goal_2x_reward_gamma_0.95_maze_16.txt'
    # vi.print_optimal_value_function(textfile = file_name)
    pygame.init()
    sarsa_l.environment.policy_simulation(sarsa_l.environment.start_state, sarsa_l.final_optimal_policy, refresh_rate=0.3)
    print("Policy Simulation Done")
    pygame.quit()
    


                    
            
    
        
        
        
        
    

