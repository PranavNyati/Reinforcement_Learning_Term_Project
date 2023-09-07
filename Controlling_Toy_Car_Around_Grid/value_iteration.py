import numpy as np
import matplotlib.pyplot as plt
import copy

from myenv import Environment

class Value_Iteration:
    def __init__(self, mdp, gamma):
        self.mdp = mdp
        self.gamma = gamma
        self.final_optimal_value_function = None
        self.final_optimal_policy = None

    def value_iteration(self, num_iterations = 1000, delta = 0.0001):
        
        self.mdp.reset()
        state_value_func = self.mdp.init_state_value_function()
        optimal_policy_func = {}
        
        for i in range(num_iterations):
            state_value_func_copy = copy.deepcopy(state_value_func)
            delta = 0.0   
                  
            for state in state_value_func.keys():
                
                # we define the environment such that from the start state, the car can only move forward 
                if (state[0], state[1]) == self.mdp.start:
                    
                    next_state, next_reward, done = self.mdp.get_next_state_and_reward(state, 1)
                    value = next_reward + self.gamma * state_value_func_copy[next_state]
                    
                    delta = max(delta, abs(value - state_value_func_copy[state]))
                    state_value_func[state] = value
                    optimal_policy_func[state] = 1
                    
                
                # elif (state[0], state[1]) != self.mdp.goal:
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
            
                # elif (state[0], state[1]) == self.mdp.goal and state[2] != 'N':
                #     optimal_policy_func[state] = 'f'
            
            if delta < 0.0001:
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
        # optimal_actions = np.((4, self.mdp.grid_size, self.mdp.grid_size), dtype=np.int32)
        #declare a 3D array to store the optimal actions
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
            
            # for cmap , let the color be 'white' for all the values
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
                    # print("hi: ", " col: ", col, " row: ", row, " i: ", i)
                    arrow = ""
                    arrow = self.mdp.directions_actions_mapping[(self.mdp.directions[i], self.mdp.actions[optimal_actions[i, row, col]])]
                    
                    text = axs.text(col, row, arrow, ha="center", va="center", color="black", fontsize=40)
                    
                    
            plt.savefig('Optimal_actions_for_direction_%s.png' % self.mdp.directions[i])
            plt.close()
                    
            
            
        
        # for i in range(4):
        #     axs[0, i].imshow(state_values[i], cmap='hot', interpolation='nearest')
        #     axs[0, i].set_title('State Values for Direction %s' % self.mdp.directions[i])
        #     axs[0, i].set_xlabel('X')
        #     axs[0, i].set_ylabel('Y')
        #     axs[0, i].set_xticks(np.arange(self.mdp.grid_size))
        #     axs[0, i].set_yticks(np.arange(self.mdp.grid_size))
        #     axs[0, i].set_xticklabels(np.arange(1, self.mdp.grid_size + 1))
        #     axs[0, i].set_yticklabels(np.arange(1, self.mdp.grid_size + 1))
        #     for j in range(self.mdp.grid_size):
        #         for k in range(self.mdp.grid_size):
        #             text = axs[0, i].text(k, j, round(state_values[i, j, k], 2), ha="center", va="center", color="w")
        #     plt.show()
        #     plt.savefig('state_values.png')
        
        return
        
        
if __name__ == "__main__":
    
    env = Environment(grid_size=12)
    vi = Value_Iteration(env, gamma = 0.99)
    vi.value_iteration()
    print("Value Iteration Done")
    file_name = 'value_iteration_start_goal_restrict_incremental_rewards_goal_reward_50x_gamma_99.txt'
    vi.print_optimal_value_function(textfile = file_name)
    
    