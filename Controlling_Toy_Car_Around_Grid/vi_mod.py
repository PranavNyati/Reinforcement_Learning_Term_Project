import numpy as np
import matplotlib.pyplot as plt
import copy
import pygame
from vi_new_env import Environment

class Value_Iteration:
    def __init__(self, mdp, gamma):
        self.mdp = mdp
        self.gamma = gamma
        self.final_optimal_value_function = None
        self.final_optimal_policy = None

    def value_iteration(self, num_iterations = 10000, prec = 0.0001):
        
        self.mdp.reset()
        state_value_func = self.mdp.init_state_value_function()
        optimal_policy_func = {}
        
        for i in range(num_iterations):
            state_value_func_copy = copy.deepcopy(state_value_func)
            delta = 0.0   
                  
            for state in state_value_func.keys():
                
                # we define the environment such that from the start state, the car can only move forward 
                # if (state[0], state[1]) == self.mdp.start:
                if state == self.mdp.start_state:
                    
                    next_state, next_reward, done = self.mdp.get_next_state_and_reward(state, 1)
                    value = next_reward + self.gamma * state_value_func_copy[next_state]
                    
                    delta = max(delta, abs(value - state_value_func_copy[state]))
                    state_value_func[state] = value
                    optimal_policy_func[state] = 1
                    
                else:
                    
                    max_value = -np.inf
                    
                    for action in self.mdp.actions.keys():
                        
                        next_state, next_reward, done = self.mdp.get_next_state_and_reward(state, action)
                        value = next_reward + self.gamma * state_value_func_copy[next_state]
                        
                        # max_value = max(max_value, value)
                        if (value > max_value):
                            max_value = value
                            optimal_action = action
                        
                    delta = max(delta, abs(max_value - state_value_func_copy[state]))
                    state_value_func[state] = max_value
                    optimal_policy_func[state] = optimal_action
            
            if delta < prec:
                print('Value iteration converged at iteration %d' % (i+1))
                break    
        
        self.final_optimal_value_function = state_value_func
        self.final_optimal_policy = optimal_policy_func
        
        return        
    
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
            
        
        for i in range(4):
            fig, axs = plt.subplots(1, 1, figsize=(20, 20))
            
            axs.imshow(state_values[i], interpolation='nearest', cmap='hot')
            axs.set_title('State Values for Direction %s' % self.mdp.directions[i])
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
            plt.savefig('State_values_for_direction_%s.png' % self.mdp.directions[i])
            plt.close()
            
        for i in range(4):
            fig, axs = plt.subplots(1, 1, figsize=(20, 20))
            color_map = np.zeros((self.mdp.grid_size, self.mdp.grid_size), dtype=np.float32)
            
            axs.imshow(optimal_actions[i], interpolation='nearest')
            axs.set_title('Optimal Actions for Direction %s' % self.mdp.directions[i])
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
                    
                    
            plt.savefig('Optimal_actions_for_direction_%s.png' % self.mdp.directions[i])
            plt.close()
                    
        return
        
        
if __name__ == "__main__":
    
    env = Environment(grid_size=16)
    vi = Value_Iteration(env, gamma = 0.95)
    vi.value_iteration()
    print("Value Iteration Done")
    file_name = 'value_iteration_forward_reward_goal_2x_reward_gamma_0.95_maze_16.txt'
    vi.print_optimal_value_function(textfile = file_name)
    pygame.init()
    vi.mdp.policy_simulation(vi.mdp.start_state, vi.final_optimal_policy, refresh_rate=0.3)
    print("Policy Simulation Done")
    pygame.quit()
    
    
    