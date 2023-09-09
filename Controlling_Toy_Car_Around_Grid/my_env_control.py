import pygame
import math
import time

class Environment_Control:
    
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
        self.actions = {0: 'f', 1: 'r', 2: 'l'}
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

        #define rewards
        self.rewards = {}
        self.rewards['hit_wall'] = -5
        self.rewards['step'] = -1
        self.rewards['goal'] = 20
        self.rewards['forward'] = 10
        
        # other parameters
        self.screen = None
        self.last_action = 'start'
        self.last_reward = 0
        self.last_done = False
        self.num_steps = 0
        self.max_steps = self.grid_size**2

    def reset(self):
        self.car_x, self.car_y = 1, self.grid_size - 2
        self.prev_car_x, self.prev_car_y = 1, self.grid_size - 2
        self.car_direction = 'N'
        self.state  = (self.car_x, self.car_y, self.car_direction)
        self.num_steps = 0
        self.state_value_func = self.init_state_value_function()
        self.last_reward = 0

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


    def get_next_state_and_reward(self, state, action_no):
        
        next_reward = 0
        next_state = None
        hit_wall = False
        prev_x = state[0]
        prev_y = state[1]
        done = False
        
        
        if self.actions[action_no] == 'f':
            if (state[2] == 'N'):
                if (self.is_on_path(state[0], state[1] - 1) == False):
                    hit_wall = True
                    next_state = state
                else:
                    next_state = (state[0], state[1] - 1, state[2])
                    
            elif (state[2] == 'S'):
                if (self.is_on_path(state[0], state[1] + 1) == False):
                    hit_wall = True
                    next_state = state
                else:
                    next_state = (state[0], state[1] + 1, state[2])
                    
            elif (state[2] == 'W'):
                if (self.is_on_path(state[0]-1, state[1]) == False):
                    hit_wall = True
                    next_state = state
                else:
                    next_state = (state[0] - 1, state[1], state[2])
            
            elif (state[2] == 'E'):
                if (self.is_on_path(state[0]+1, state[1]) == False):
                    hit_wall = True
                    next_state = state
                else:
                    next_state = (state[0] + 1, state[1], state[2])
                    
                
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
                
        
        # REWARDS FOR ALL POSSIBLE STATE ACITON PAIRS:
        
        # Reward for reaching the goal
        if (next_state[0], next_state[1]) == self.goal and (prev_x, prev_y) == (self.goal[0] + 1, self.goal[1]):
            done = True
            # print("Goal reached!")
            next_reward =  self.rewards['goal']

        # Reward for moving forward in some direction
        elif (next_state[0], next_state[1]) != (prev_x, prev_y):
            next_reward = self.rewards['forward']
        
        elif hit_wall:
            next_reward = self.rewards['hit_wall']
        
        else:
            next_reward = self.rewards['step']
              
        
        return next_state, next_reward, done
              
    def policy_simulation(self, start_state, policy, refresh_rate=0.5):
        
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption('RL Car Environment Visualization')
        
        self.reset()
        self.state = start_state
        self.render()
        done_flag = False
        ctr = 0
        
        while(ctr < self.max_steps):
            
            next_state, reward, done = self.get_next_state_and_reward(self.state, policy[self.state])
            self.state = next_state
            self.last_action = self.actions[policy[self.state]]
            self.last_reward = reward
            self.last_done = done
            
            self._update_visualization()
            time.sleep(refresh_rate)  # Delay for visualization 
            self.render() 
            ctr += 1
            
            if done:
                done_flag = True
                print("Goal reached successfully within {} steps!".format(ctr))
                break
            
        if done_flag == False:
            print("Goal could not reached using the given policy within {} steps (max steps allowed)!".format(ctr))
            


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
                
                # # To know the reference point (0, 0) => filled with red
                # if (row == 0 and col == 0):
                #     pygame.draw.rect(self.screen, (255, 0, 0), (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))
                
                if ((row == 1 or row == self.grid_size-2) and (col <= self.grid_size - 2  and col >= 1 )) or ((col == 1 or col == self.grid_size-2) and (row <= self.grid_size -2  and row >= 1 )):
                    
                    # to mark the source as well as the destination (both are same) => filled with green
                    if (row == self.grid_size-2  and col == 1):
                        pygame.draw.rect(self.screen, (0, 255, 0), (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))

                    else:
                        pygame.draw.rect(self.screen, (139, 69, 19), (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))

        car_center = (self.state[0] * self.cell_size + self.cell_size // 2, self.state[1] * self.cell_size + self.cell_size // 2)
        car_side_length = self.cell_size // 2
        if self.state[2] == 'N':
            car_points = [(car_center[0], car_center[1] - car_side_length),
                          (car_center[0] - car_side_length, car_center[1] + car_side_length),
                          (car_center[0] + car_side_length, car_center[1] + car_side_length)]
        elif self.state[2] == 'S':
            car_points = [(car_center[0], car_center[1] + car_side_length),
                          (car_center[0] - car_side_length, car_center[1] - car_side_length),
                          (car_center[0] + car_side_length, car_center[1] - car_side_length)]
        elif self.state[2] == 'W':
            car_points = [(car_center[0] - car_side_length, car_center[1]),
                          (car_center[0] + car_side_length, car_center[1] - car_side_length),
                          (car_center[0] + car_side_length, car_center[1] + car_side_length)]
        elif self.state[2] == 'E':
            car_points = [(car_center[0] + car_side_length, car_center[1]),
                          (car_center[0] - car_side_length, car_center[1] - car_side_length),
                          (car_center[0] - car_side_length, car_center[1] + car_side_length)]

        pygame.draw.polygon(self.screen, (0, 0, 0), car_points)

        # Draw information box
        info_box = pygame.Surface((self.window_size[0], 50))
        info_box.fill((200, 200, 200))
        info_text = f"State: ({self.state[0]}, {self.state[1]}, {self.state[2]});  Action: {self.last_action};  Reward: {self.last_reward};  Done: {self.last_done}"
        font = pygame.font.Font(None, 24)
        text = font.render(info_text, True, (0, 0, 0))
        info_box.blit(text, (10, 10))
        self.screen.blit(info_box, (0, self.window_size[1] - 50))

        pygame.display.flip()

