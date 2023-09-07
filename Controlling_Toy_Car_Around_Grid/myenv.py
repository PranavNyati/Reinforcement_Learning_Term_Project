import pygame
import math
import time

class Environment:
    def __init__(self, grid_size=18, cell_size=50):
        
        # Environment parameters
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.window_size = (grid_size * cell_size, grid_size * cell_size+50)
        
        # initial state and direction of the car
        self.car_x, self.car_y = 1, grid_size - 2
        self.prev_car_x, self.prev_car_y = 1, grid_size - 2
        self.start = (1, grid_size - 2)
        self.car_direction = 'N'
        self.state  = (self.car_x, self.car_y, self.car_direction)
        self.start_state = self.state
        
        #define actions
        self.actions = {1: 'f', 2: 'r', 3: 'l'}
        self.directions = ['N', 'S', 'W', 'E']
        # self.directions_dict = {0:'N', 1:'S', 2:'W', 3:'E'}
        
        # direction to action mapping
        self.directions_actions_mapping = {}
        self.directions_actions_mapping[('N', 'f')] = '\u2191'
        self.directions_actions_mapping[('S', 'f')] = '\u2193'
        self.directions_actions_mapping[('W', 'f')] = '\u2190'
        self.directions_actions_mapping[('E', 'f')] = '\u2192'
        self.directions_actions_mapping[('N', 'r')] = '\u2192'
        self.directions_actions_mapping[('S', 'r')] = '\u2190'
        self.directions_actions_mapping[('W', 'r')] = '\u2191'
        self.directions_actions_mapping[('E', 'r')] = '\u2193'
        self.directions_actions_mapping[('N', 'l')] = '\u2190'
        self.directions_actions_mapping[('S', 'l')] = '\u2192'    
        self.directions_actions_mapping[('W', 'l')] = '\u2193'
        self.directions_actions_mapping[('E', 'l')] = '\u2191'
                          
        # state space is defined as (x, y)
        self.num_states = 4*grid_size - 12
        self.state_value_func = self.init_state_value_function()
        
        # goal is to come back to the starting point
        self.goal = (1, grid_size - 2)
        self.landmarks = [(1, (grid_size-2)//2), (1, 1), ((grid_size-2)//2, 1), (grid_size-2, 1), (grid_size-2, (grid_size-2)//2), (grid_size-2, grid_size-2), ((grid_size-2)//2, grid_size-2)]
    
        #define rewards
        self.rewards = {}
        self.rewards['hit_wall'] = -10
        self.rewards['step'] = -1
        self.rewards['goal'] = 50*self.grid_size
        self.rewards['landmark'] = [self.grid_size, 2*self.grid_size, 3*self.grid_size, 4*self.grid_size, 5*self.grid_size, 6*self.grid_size, 7*self.grid_size]
        
        
        # self.corner_states = [(1, 1), (grid_size - 2, 1), (1, grid_size - 2), (grid_size - 2, grid_size - 2)]
        # self.corners_visited = {}
        # for corner in self.corner_states:
        #     self.corners_visited[corner] = False
                
        # self.screen = pygame.display.set_mode(self.window_size)
        # pygame.display.set_caption('RL Car Environment Visualization')
        
        self.last_action = 'start'
        self.last_reward = 0
        self.last_done = False
        self.num_steps = 0
        self.max_steps = 200

    def reset(self):
        self.car_x, self.car_y = 1, self.grid_size - 2
        self.prev_car_x, self.prev_car_y = 1, self.grid_size - 2
        self.car_direction = 'N'
        self.state  = (self.car_x, self.car_y, self.car_direction)
        self.num_steps = 0
        self.state_value_func = self.init_state_value_function()
        self.last_reward = 0
        # for corner in self.corner_states:
        #     self.corners_visited[corner] = False

    def init_state_value_function(self):
        state_value_dict = {}
        for row in [1, self.grid_size-2]:
            for col in range(1, self.grid_size-1):
                for direction in self.directions:
                    state_value_dict[(col, row, direction)] = 0.0
        
        for col in [1, self.grid_size-2]:
            for row in range(1, self.grid_size-1):
                for direction in self.directions:
                    state_value_dict[(col, row, direction)] = 0.0
        
        return state_value_dict

    def is_on_path(self, car_x, car_y):
        return (((car_y == 1 or car_y == self.grid_size-2) and (car_x <= self.grid_size - 2  and car_x >= 1 )) or ((car_x == 1 or car_x == self.grid_size-2) and (car_y <= self.grid_size -2  and car_y >= 1 )))

    # def check_corner_visited(self):
    #     for corner in self.corners_visited.keys():
    #         if self.corners_visited[corner] == False:
    #             return False
    #     return True

    # def get_next_state_and_reward(self, state, action_no):
        
    #     next_reward = self.rewards['step']
    #     next_state = None
    #     hit_wall = False
    #     prev_x = state[0]
    #     prev_y = state[1]
    #     done = False
        
    #     # isOnPath = self.is_on_path(state[0], state[1])
        
    #     if (state[0], state[1]) != self.goal or state == self.start_state:
    #         if self.actions[action_no] == 'f':
    #             if (state[2] == 'N'):
    #                 if (self.is_on_path(state[0], state[1] - 1) == False):
    #                     hit_wall = True
    #                     next_state = (state[0], state[1], state[2])
    #                 else:
    #                     # prev_y = state[1]
    #                     next_state = (state[0], state[1] - 1, state[2])
                        
    #             elif (state[2] == 'S'):
    #                 if (self.is_on_path(state[0], state[1] + 1) == False):
    #                     hit_wall = True
    #                     next_state = (state[0], state[1], state[2])
    #                 else:
    #                     next_state = (state[0], state[1] + 1, state[2])
                        
    #             elif (state[2] == 'W'):
    #                 if (self.is_on_path(state[0]-1, state[1]) == False):
    #                     hit_wall = True
    #                     next_state = (state[0], state[1], state[2])
    #                 else:
    #                     next_state = (state[0] - 1, state[1], state[2])
                
    #             elif (state[2] == 'E'):
    #                 if (self.is_on_path(state[0]+1, state[1]) == False):
    #                     hit_wall = True
    #                     next_state = (state[0], state[1], state[2])
    #                 else:
    #                     next_state = (state[0] + 1, state[1], state[2])
                        
    #             # elif (state[2] == 'S'):
    #             #     if (self.car_y + 1 > self.grid_size - 2):
    #             #         hit_wall = True
    #             #     prev_y = state[1]
    #             #     next_state = (state[0], min(state[1] + 1, self.grid_size - 2), state[2])
                    
    #         elif self.actions[action_no] == 'r':
                
    #             if state[2] == 'N':
    #                 next_state = (state[0], state[1], 'E')
    #             elif state[2] == 'S':
    #                 next_state = (state[0], state[1], 'W')
    #             elif state[2] == 'W':
    #                 next_state = (state[0], state[1], 'N')
    #             elif state[2] == 'E':
    #                 next_state = (state[0], state[1], 'S')
                    
    #         elif self.actions[action_no] == 'l':

    #             if state[2] == 'N':
    #                 next_state = (state[0], state[1], 'W')
    #             elif state[2] == 'S':
    #                 next_state = (state[0], state[1], 'E')
    #             elif state[2] == 'W':
    #                 next_state = (state[0], state[1], 'S')
    #             elif state[2] == 'E':
    #                 next_state = (state[0], state[1], 'N')
                    
    #     elif (state[0], state[1]) == self.goal and state[2] != 'N':
    #         next_state = state
                
    #     # if (next_state[0], next_state[1]) in self.corner_states and self.corners_visited[(next_state[0], next_state[1])] == False:
    #     #     self.corners_visited[(next_state[0], next_state[1])] = True
                
    #     # rewards for the particular state and action
    #     # if (next_state[0], next_state[1]) == self.goal and (prev_x, prev_y) != self.goal and self.check_corner_visited() == True:
    #     #     done = True
    #     #     # print("Goal reached!")
    #     #     next_reward = self.rewards['goal_reward']
    #     # elif (next_state[0], next_state[1]) == self.sub_goal and (prev_x, prev_y) != self.sub_goal:
    #     #     next_reward = self.rewards['sub_goal_reward']
    #     # elif hit_wall:
    #     #     next_reward = self.rewards['hit_wall_reward']  
    #     # else:
    #     #     next_reward = self.rewards['step_reward']
            
        
    #     # REWARDS FOR ALL POSSIBLE STATE ACITON PAIRS:
        
    #     # Reward for reaching the goal
    #     if (next_state[0], next_state[1]) == self.goal and (prev_x, prev_y) != self.goal and (prev_x, prev_y) == (self.goal[0] + 1, self.goal[1]):
    #         done = True
    #         print("Goal reached!")
    #         next_reward = next_reward + self.rewards['goal']
            
    #     # Reward for reaching the sub-goals/landmarks
    #     elif (next_state[0], next_state[1]) in self.landmarks and (prev_x, prev_y) != (next_state[0], next_state[1]):
    #         next_reward = next_reward + self.rewards['landmark'][self.landmarks.index((next_state[0], next_state[1]))]
            
    #         # if (next_state[0], next_state[1]) != self.landmarks[6]:
    #         #     next_reward = next_reward + self.rewards['landmark'][self.landmarks.index((next_state[0], next_state[1]))]
    #         # else:
    #         #     if (prev_x, prev_y) == (self.landmarks[6][0] - 1, self.landmarks[6][1]):
    #         #         next_reward = next_reward + self.rewards['landmark'][self.landmarks.index((next_state[0], next_state[1]))]
            
            
    #     # Reward for hitting the wall
    #     elif hit_wall:
    #         next_reward = next_reward + self.rewards['hit_wall']
            
            
    #     return next_state, next_reward, done
     

    def get_next_state_and_reward(self, state, action_no):
        
        next_reward = self.rewards['step']
        next_state = None
        hit_wall = False
        prev_x = state[0]
        prev_y = state[1]
        done = False
        
        # isOnPath = self.is_on_path(state[0], state[1])
        
        if self.actions[action_no] == 'f':
            if (state[2] == 'N'):
                if (self.is_on_path(state[0], state[1] - 1) == False):
                    hit_wall = True
                    next_state = (state[0], state[1], state[2])
                else:
                    # prev_y = state[1]
                    next_state = (state[0], state[1] - 1, state[2])
                    
            elif (state[2] == 'S'):
                if (self.is_on_path(state[0], state[1] + 1) == False):
                    hit_wall = True
                    next_state = (state[0], state[1], state[2])
                else:
                    next_state = (state[0], state[1] + 1, state[2])
                    
            elif (state[2] == 'W'):
                if (self.is_on_path(state[0]-1, state[1]) == False):
                    hit_wall = True
                    next_state = (state[0], state[1], state[2])
                else:
                    next_state = (state[0] - 1, state[1], state[2])
            
            elif (state[2] == 'E'):
                if (self.is_on_path(state[0]+1, state[1]) == False):
                    hit_wall = True
                    next_state = (state[0], state[1], state[2])
                else:
                    next_state = (state[0] + 1, state[1], state[2])
                    
            # elif (state[2] == 'S'):
            #     if (self.car_y + 1 > self.grid_size - 2):
            #         hit_wall = True
            #     prev_y = state[1]
            #     next_state = (state[0], min(state[1] + 1, self.grid_size - 2), state[2])
                
        elif self.actions[action_no] == 'r':
            
            if state[2] == 'N':
                next_state = (state[0], state[1], 'E')
            elif state[2] == 'S':
                next_state = (state[0], state[1], 'W')
            elif state[2] == 'W':
                next_state = (state[0], state[1], 'N')
            elif state[2] == 'E':
                next_state = (state[0], state[1], 'S')
                
        elif self.actions[action_no] == 'l':

            if state[2] == 'N':
                next_state = (state[0], state[1], 'W')
            elif state[2] == 'S':
                next_state = (state[0], state[1], 'E')
            elif state[2] == 'W':
                next_state = (state[0], state[1], 'S')
            elif state[2] == 'E':
                next_state = (state[0], state[1], 'N')
                
        # if (next_state[0], next_state[1]) in self.corner_states and self.corners_visited[(next_state[0], next_state[1])] == False:
        #     self.corners_visited[(next_state[0], next_state[1])] = True
        
        # REWARDS FOR ALL POSSIBLE STATE ACITON PAIRS:
        
        # Reward for reaching the goal
        if (next_state[0], next_state[1]) == self.goal and (prev_x, prev_y) != self.goal and (prev_x, prev_y) == (self.goal[0] + 1, self.goal[1]):
            done = True
            print("Goal reached!")
            next_reward = next_reward + self.rewards['goal']
            
        # Reward for reaching the sub-goals/landmarks
        elif (next_state[0], next_state[1]) in self.landmarks and (prev_x, prev_y) != (next_state[0], next_state[1]):
            next_reward = next_reward + self.rewards['landmark'][self.landmarks.index((next_state[0], next_state[1]))]
            
        # Reward for hitting the wall
        elif hit_wall:
            next_reward = next_reward + self.rewards['hit_wall']
            
        return next_state, next_reward, done
                

    def step(self, action):
        self.last_action = action
        hit_wall = False
        if action == 'f':
            if self.car_direction == 'N':
                if (self.car_y - 1 < 1):
                    hit_wall = True
                self.prev_car_y = self.car_y
                self.car_y = max(self.car_y - 1, 1)
                
                
            elif self.car_direction == 'S':
                if (self.car_y + 1 > self.grid_size - 2):
                    hit_wall = True
                self.prev_car_y = self.car_y
                self.car_y = min(self.car_y + 1, self.grid_size - 2)
                
            elif self.car_direction == 'W':
                if (self.car_x - 1 < 1):
                    hit_wall = True
                self.prev_car_x = self.car_x
                self.car_x = max(self.car_x - 1, 1)
                
            elif self.car_direction == 'E':
                if (self.car_x + 1 > self.grid_size - 2):
                    hit_wall = True
                self.prev_car_x = self.car_x
                self.car_x = min(self.car_x + 1, self.grid_size - 2)

        elif action == 'r':
            self.prev_car_x = self.car_x
            self.prev_car_y = self.car_y
            
            if self.car_direction == 'N':
                self.car_direction = 'E'
            elif self.car_direction == 'S':
                self.car_direction = 'W'
            elif self.car_direction == 'W':
                self.car_direction = 'N'
            elif self.car_direction == 'E':
                self.car_direction = 'S'
        
        elif action == 'l':
            self.prev_car_x = self.car_x
            self.prev_car_y = self.car_y
            
            if self.car_direction == 'N':
                self.car_direction = 'W'
            elif self.car_direction == 'S':
                self.car_direction = 'E'
            elif self.car_direction == 'W':
                self.car_direction = 'S'
            elif self.car_direction == 'E':
                self.car_direction = 'N'
        
        # isOnPath = (self.car_y == 0 and self.car_x == self.grid_size - 1) or (self.car_x == self.grid_size - 2) or (self.car_y == self.grid_size - 1 and self.car_x != self.grid_size - 1)
        isOnPath = self.is_on_path(self.car_x, self.car_y)
        
        
        if (self.car_x, self.car_y) == self.goal and (self.prev_car_x, self.prev_car_y) != self.goal:
            reward = self.rewards['goal_reward']
            done = True
        
        elif (self.car_x, self.car_y) == self.sub_goal and self.num_steps > 0:
            reward = self
            done = False    

        elif hit_wall:
            reward = self.rewards['hit_wall_reward']
            done = False
            
        else: # penalize the agent for each step taken
            reward = self.rewards['step_reward']
            done = False

        self.last_reward = reward
        self.last_done = done
        self.num_steps += 1
        
        if (self.num_steps >= self.max_steps):
            done = True

        self._update_visualization()
        time.sleep(1)  # Delay for visualization

        return (self.car_x, self.car_y, self.car_direction), reward, done, {}

    def render(self):
        self._update_visualization()

    def _update_visualization(self):
        self.screen.fill((255, 255, 255))
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                pygame.draw.rect(self.screen, (230, 230, 250), (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size), 1)
                
                # To know the reference point (0, 0) => filled with red
                if (row == 0 and col == 0):
                    pygame.draw.rect(self.screen, (255, 0, 0), (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))
                
                if ((row == 1 or row == self.grid_size-2) and (col <= self.grid_size - 2  and col >= 1 )) or ((col == 1 or col == self.grid_size-2) and (row <= self.grid_size -2  and row >= 1 )):
                    
                    # to mark the source as well as the destination (both are same) => filled with green
                    if (row == self.grid_size-2  and col == 1):
                        pygame.draw.rect(self.screen, (0, 255, 0), (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))

                    else:
                        pygame.draw.rect(self.screen, (139, 69, 19), (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))

        car_center = (self.car_x * self.cell_size + self.cell_size // 2, self.car_y * self.cell_size + self.cell_size // 2)
        car_side_length = self.cell_size // 2
        if self.car_direction == 'N':
            car_points = [(car_center[0], car_center[1] - car_side_length),
                          (car_center[0] - car_side_length, car_center[1] + car_side_length),
                          (car_center[0] + car_side_length, car_center[1] + car_side_length)]
        elif self.car_direction == 'S':
            car_points = [(car_center[0], car_center[1] + car_side_length),
                          (car_center[0] - car_side_length, car_center[1] - car_side_length),
                          (car_center[0] + car_side_length, car_center[1] - car_side_length)]
        elif self.car_direction == 'W':
            car_points = [(car_center[0] - car_side_length, car_center[1]),
                          (car_center[0] + car_side_length, car_center[1] - car_side_length),
                          (car_center[0] + car_side_length, car_center[1] + car_side_length)]
        elif self.car_direction == 'E':
            car_points = [(car_center[0] + car_side_length, car_center[1]),
                          (car_center[0] - car_side_length, car_center[1] - car_side_length),
                          (car_center[0] - car_side_length, car_center[1] + car_side_length)]

        pygame.draw.polygon(self.screen, (0, 0, 0), car_points)

        # Draw information box
        info_box = pygame.Surface((self.window_size[0], 50))
        info_box.fill((200, 200, 200))
        info_text = f"State: ({self.car_x}, {self.car_y}, {self.car_direction});  Action: {self.last_action};  Reward: {self.last_reward};  Done: {self.last_done}"
        font = pygame.font.Font(None, 24)
        text = font.render(info_text, True, (0, 0, 0))
        info_box.blit(text, (10, 10))
        self.screen.blit(info_box, (0, self.window_size[1] - 50))

        pygame.display.flip()

# Initialize Pygame
# pygame.init()

# # Create environment instance
# env = Environment()

# # Example of using the environment
# state = env.reset()
# env.render()
# for _ in range(2):
#     action = 'f'
#     next_state, reward, done, _ = env.step(action)
#     env.render()

# next_state, reward, done, _ = env.step('r')
# env.render()

# for _ in range(100):
#     action = 'f'
#     next_state, reward, done, _ = env.step(action)
#     env.render()

# # Quit Pygame
# pygame.quit()

