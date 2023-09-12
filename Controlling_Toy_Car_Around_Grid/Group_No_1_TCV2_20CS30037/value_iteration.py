# Group Number: 1
# Roll Number (Name of member): 20CS30037 (Pranav Nyati)
# Project Code: TCv2
# Project Title: Controlling a Toy Car around a Grid [Version 2]

import numpy as np
import matplotlib.pyplot as plt
import copy
import pygame
from my_env_control import Environment_Control

class Value_Iteration:
    def __init__(self, mdp, gamma):
        self.mdp = mdp # the MDP environment for which we want to learn the optimal policy
        self.gamma = gamma # discount factor
        self.final_optimal_value_function = None
        self.final_optimal_policy = None
        self.deltas = None

    def value_iteration(self, num_iterations = 1000, prec = 0.0001):
        
        self.mdp.reset() # reset the environment to the start state
        state_value_func = self.mdp.init_state_value_function() # initialize the state value function to 0 for all states
        optimal_policy_func = {}
        
        self.deltas = []
        
        for i in range(num_iterations):
            state_value_func_copy = copy.deepcopy(state_value_func)
            delta = 0.0   
                  
            for state in state_value_func.keys():
                # we define the environment such that from the start state, the car can only move forward (action = 0)
                if state == self.mdp.start_state:
                    
                    next_state, next_reward, done = self.mdp.get_next_state_and_reward(state, 0)
                    value = next_reward + self.gamma * state_value_func_copy[next_state]
                    
                    delta = max(delta, abs(value - state_value_func_copy[state]))
                    state_value_func[state] = value
                    optimal_policy_func[state] = 0
                    
                else:
                    
                    # update the value function for each state by taking the max over all possible actions (using the Bellman optimality equation)
                    
                    max_value = -np.inf
                    
                    for action in self.mdp.actions.keys():
                        
                        next_state, next_reward, done = self.mdp.get_next_state_and_reward(state, action)
                        value = next_reward + self.gamma * state_value_func_copy[next_state]
                        
                        if (value > max_value):
                            max_value = value
                            optimal_action = action
                        
                    delta = max(delta, abs(max_value - state_value_func_copy[state])) # calculate the max change in value function of any state
                    state_value_func[state] = max_value
                    optimal_policy_func[state] = optimal_action
                    
            self.deltas.append(delta)
            
            if delta < prec: # if the max change in value function of any state is less than the precision, we have converged
                print('Value iteration converged at iteration %d' % (i+1))
                break    
        
        self.final_optimal_value_function = state_value_func
        self.final_optimal_policy = optimal_policy_func
        
        return  self.final_optimal_value_function, self.final_optimal_policy
    
    def print_optimal_value_function(self, textfile = None):
        
        if self.final_optimal_value_function is None:
            print('Value function not computed yet')
            return
        
        if (self.mdp is not None and textfile is None):
            for state in self.final_optimal_value_function.keys():
                print('State: %s, Optimal Value: %f' % (state, self.final_optimal_value_function[state]))
        
        elif (self.mdp is not None and textfile is not None):
            f = open(textfile, 'w')
            for state in self.final_optimal_value_function.keys():
                f.write('State: %s, Optimal Value: %f' % (state, self.final_optimal_value_function[state]))
                f.write('\n')
                f.write('State: %s, Optimal Action: %s' % (state, self.final_optimal_policy[state]))
                f.write('\n\n')
            f.close()
        
        
        state_values = np.zeros((4, self.mdp.grid_size, self.mdp.grid_size), dtype=np.float32)
        optimal_actions = np.ones((4, self.mdp.grid_size, self.mdp.grid_size), dtype=np.int32)
        
        for state in self.final_optimal_value_function.keys():
            state_values[self.mdp.directions.index(state[2]), state[1], state[0]] = self.final_optimal_value_function[state]
            optimal_actions[self.mdp.directions.index(state[2]), state[1], state[0]] = self.final_optimal_policy[state]
            
        # Plot the optimal state values for each state learnt using Value Iteration
        for i in range(4):
            fig, axs = plt.subplots(1, 1, figsize=(20, 20))
            
            axs.imshow(state_values[i], interpolation='nearest', cmap='hot')
            axs.set_title('VI: State Values for Direction %s' % self.mdp.directions[i])
            axs.set_xlabel('X')
            axs.set_ylabel('Y')
            axs.set_xticks(np.arange(self.mdp.grid_size))
            axs.set_yticks(np.arange(self.mdp.grid_size))
            axs.set_xticklabels(np.arange(1, self.mdp.grid_size + 1))
            axs.set_yticklabels(np.arange(1, self.mdp.grid_size + 1))
            for row in range(self.mdp.grid_size):
                for col in range(self.mdp.grid_size):
                    # text = axs.text(j, k, round(state_values[i, j, k], 2), ha="center", va="center", color="blue", fontsize=15)
                    text = axs.text(col, row, round(state_values[i, row, col], 2), ha="center", va="center", color="blue", fontsize=15)
                    
            # plt.show()
            plt.savefig('VI_State_values_for_direction_%s.png' % self.mdp.directions[i])
            plt.close()
            
            
        # Plot the optimal actions for each state learnt using Value Iteration  
        for i in range(4):
            fig, axs = plt.subplots(1, 1, figsize=(20, 20))
            color_map = np.zeros((self.mdp.grid_size, self.mdp.grid_size), dtype=np.float32)
            
            axs.imshow(optimal_actions[i], interpolation='nearest')
            axs.set_title('VI: Optimal Actions for Direction %s' % self.mdp.directions[i])
            axs.set_xlabel('X')
            axs.set_ylabel('Y')
            axs.set_xticks(np.arange(self.mdp.grid_size))
            axs.set_yticks(np.arange(self.mdp.grid_size))
            axs.set_xticklabels(np.arange(1, self.mdp.grid_size + 1))
            axs.set_yticklabels(np.arange(1, self.mdp.grid_size + 1))
            for row in range(self.mdp.grid_size):
                for col in range(self.mdp.grid_size):

                    arrow = ""
                    arrow = self.mdp.directions_actions_mapping[(self.mdp.directions[i], self.mdp.actions[optimal_actions[i, row, col]])]
                    text = axs.text(col, row, arrow, ha="center", va="center", color="black", fontsize=40)
                    
                    
            plt.savefig('VI_Optimal_actions_for_direction_%s.png' % self.mdp.directions[i])
            plt.close()
            
            
        # Plot how the max delta changes with each iteration
        plt.figure(figsize=(10, 5))
        plt.plot(self.deltas)
        plt.title("VI: Max change in value function of any state across iterations")
        plt.xlabel("Iteration Number")
        plt.ylabel("Max change in value function of any state")
        plt.grid(True)
        plt.savefig("Convergence of Value Iteration across iterations for gamma = %f_maze_16.png" % self.gamma)  
        plt.close()
                          
        return
        
        
if __name__ == "__main__":
    
    env = Environment_Control(grid_size=16) # initialize the environment
    vi = Value_Iteration(env, gamma = 0.95) # initialize the Value Iteration algorithm
    vi.value_iteration(num_iterations = 2000, prec = 0.0001)
    print("Value Iteration Done")
    file_name = 'VI_forward_reward_goal_2x_reward_gamma_1_maze_16.txt'
    vi.print_optimal_value_function(textfile = file_name)
    
    # simulate the optimal policy learnt using Value Iteration
    pygame.init()
    vi.mdp.policy_simulation(vi.mdp.start_state, vi.final_optimal_policy, refresh_rate=0.1)
    print("Policy Simulation Done")
    pygame.quit()
    
    
    