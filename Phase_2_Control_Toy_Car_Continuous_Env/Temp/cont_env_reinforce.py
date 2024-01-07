import pygame
import math
import numpy as np

class ContinuousCarRadarEnv:
    def __init__(self, window_size=(1000, 1050)):
        self.window_size = window_size

        # Pygame initialization
        # pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption('Continuous Car Control with Radar')

        # Colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.brown = (139, 69, 19)
        self.dark_green = (0, 100, 0)

        # Circular path parameters
        self.center = (self.window_size[0] // 2, self.window_size[1] // 2 - 50)
        self.radius = 300
        self.path_thickness = 80
        self.no_path_radius = 140
        self.no_path_outer_radius = self.radius + self.path_thickness

        # Stochastic wind parameters
        self.wind_mean_x = 0.3
        self.wind_variance_x = 0.05
        self.wind_speed_x = np.random.normal(self.wind_mean_x, self.wind_variance_x)
        
        self.wind_mean_y = 0.3
        self.wind_variance_y = 0.05
        self.wind_speed_y = np.random.normal(self.wind_mean_y, self.wind_variance_y)

        # Car parameters
        self.car_length = 30
        self.car_width = 20
        self.car_speed = 1.2
        self.car_angle = math.pi / 2  # Start facing upward
        self.car_x, self.car_y = self.center[0] + self.radius - (self.path_thickness//2), self.center[1]
        self.start_pos_x, self.start_pos_y = self.car_x, self.car_y
        self.half_lap = False
        self.three_quarter_lap = False
        self.no_access_flag = False

        # Radar parameters
        self.num_rays = 5
        self.ray_lengths = np.array([50, 60, 70, 60, 50])
        self.ray_angles = np.array([-math.pi / 4, -math.pi / 8, 0, math.pi / 8, math.pi / 4])
        self.ray_values = np.array([0] * self.num_rays)  # Store radar ray values (1, -1, or 0)
        self.ray_x = np.array([0] * self.num_rays)
        self.ray_y = np.array([0] * self.num_rays)

        # State parameters and action space
        self.state_dim = self.num_rays + 3
        self.action_dim = 3

        # Rewards and score
        self.reward_inside_path = 2
        self.reward_outside_path = -200
        self.score = 0
        self.time_penalty = -1
        self.angle_change_penalty = -10
        self.goal_reward = 5000
        self.three_quarter_lap_reward = 1500
        self.no_access_hit_penalty = -2000
        self.obstacle_sense_penalty = -5

        # Initialize Pygame clock
        self.clock = pygame.time.Clock()

        # Initialize the environment
        self.reset()

    def step(self, action):
        # Take an action (change car angle)
        if action == 0:  # Steer left
            self.car_angle += 0.1
        elif action == 1:  # Go straight
            pass
        elif action == 2:  # Steer right
            self.car_angle -= 0.1
            
        reward = 0

        # sample wind speed from a normal distribution
        self.wind_speed_x = np.random.normal(self.wind_mean_x, self.wind_variance_x)
        self.wind_speed_y = np.random.normal(self.wind_mean_y, self.wind_variance_y)
        # print("Wind speed x: ", self.wind_speed_x)
        # print("Wind speed y: ", self.wind_speed_y)

        # Calculate new car position
        self.car_x += self.car_speed * math.cos(self.car_angle) + self.wind_speed_x
        self.car_y -= self.car_speed * math.sin(self.car_angle)
        self.car_y += self.wind_speed_y

        # Wrap car position around the circular path
        self.car_x %= self.window_size[0]
        self.car_y %= self.window_size[1]
        
        distance_to_center = math.sqrt((self.car_x - self.center[0]) ** 2 + (self.car_y - self.center[1]) ** 2)
        
        no_access = False
        
        # Also draw an inner circle within the circular track, where the car cannot go , reflect the car back (Wrap car position around the inner black circle)
        if (distance_to_center < self.no_path_radius):
            
            # reflect the car back
            self.car_x -= self.car_speed * math.cos(self.car_angle) + self.wind_speed_x + np.random.uniform(0, 0.5)
            self.car_y += self.car_speed * math.sin(self.car_angle) + np.random.uniform(0, 0.5)
            self.car_y -= self.wind_speed_y  
            self.car_angle += math.pi  
            reward += self.no_access_hit_penalty
            # no_access = True
            
        if (distance_to_center > self.no_path_outer_radius):
                
            # reflect the car back
            self.car_x -= self.car_speed * math.cos(self.car_angle) + self.wind_speed_x + np.random.uniform(0, 0.5)
            self.car_y += self.car_speed * math.sin(self.car_angle) + np.random.uniform(0, 0.5)
            self.car_y -= self.wind_speed_y  
            self.car_angle += math.pi  
            reward += self.no_access_hit_penalty
            # no_access = True
        
        

        # # Cast radar rays and update ray_values
        # for i in range(self.num_rays):
        #     ray_x = self.car_x + self.ray_lengths[i] * math.cos(self.car_angle + self.ray_angles[i])
        #     ray_y = self.car_y - self.ray_lengths[i] * math.sin(self.car_angle + self.ray_angles[i])

        #     distance_to_center = math.sqrt((ray_x - self.center[0])**2 + (ray_y - self.center[1])**2)
        #     if (self.radius - self.path_thickness < distance_to_center < self.radius):
        #         self.ray_values[i] = 1
        #     else:
        #         self.ray_values[i] = -1

        if no_access == False:
            # parallelize the code for ray casting
            self.ray_x = self.car_x + self.ray_lengths * np.cos(self.car_angle + self.ray_angles)
            self.ray_y = self.car_y - self.ray_lengths * np.sin(self.car_angle + self.ray_angles)
            distance_to_center = np.sqrt((self.ray_x - self.center[0])**2 + (self.ray_y - self.center[1])**2)
            self.ray_values = np.where((self.radius - self.path_thickness < distance_to_center) & (distance_to_center < self.radius), 1, -1)
            
            # Update score based on radar ray values, incurring a penalty if majority of the rays are -1
            if np.where(self.ray_values == -1)[0].shape[0] > (self.num_rays//2):
                reward += self.obstacle_sense_penalty
            
            
            # Check if the car is inside the circular path
            distance_to_center = math.sqrt((self.car_x - self.center[0]) ** 2 + (self.car_y - self.center[1]) ** 2)
            if (self.radius - self.path_thickness < distance_to_center < self.radius):
                reward += self.reward_inside_path
            else:
                reward += self.reward_outside_path
                
            # Test if the car has completed half a lap, then set the flag to true
            if self.half_lap == False:
                if (self.start_pos_y - 1 < self.car_y < self.start_pos_y + 2) and (self.center[0] - self.radius < self.car_x < self.center[0] - self.radius + self.path_thickness):
                    self.half_lap = True
                    print("Half lap completed!")
                    
            # Test if the car has completed three quarter lap, then set the flag to true and give a reward (to encourage the car to complete the lap in right direction)
            if (self.half_lap == True) and (self.three_quarter_lap == False):
                if (self.start_pos_y + self.radius - self.path_thickness < self.car_y < self.start_pos_y + self.radius) and (self.centre[0] -1 < self.car_x < self.centre[0] + 2):
                    self.three_quarter_lap = True
                    print("Three quarter lap completed!")
                    reward += self.three_quarter_lap_reward
                
            
            # add a penalty for each time step
            reward += self.time_penalty
            
            # add a penalty for changing the angle of the car, so that the car learns to maneuver smoothly, rather than zig-zagging
            reward += self.angle_change_penalty * abs(action - 1)
            

            # Check if the car has reached the goal, then set done to True and give a big reward
            done = False
            if self.half_lap == True and self.three_quarter_lap == True:
                
                if (self.start_pos_y - 2 < self.car_y < self.start_pos_y + 1) and (self.center[0] + self.radius - self.path_thickness < self.car_x < self.center[0] + self.radius) :
                # and (0 < self.car_angle < math.pi)
                    done = True
                    reward += self.goal_reward
        
        else:
            done = True
            self.no_access_flag = True
            reward += self.time_penalty
            
            

        self.score = reward
        
        
        # Return observation, reward, done, info
        observation = np.array(self.ray_values.tolist() + [self.car_x, self.car_y, self.car_angle])
        return observation, reward, done, {}

    def reset(self):
        # Reset car position and radar values
        self.car_x, self.car_y =  self.center[0] + self.radius - (self.path_thickness//2), self.center[1]
        self.car_angle = math.pi / 2
        self.ray_values = np.array([0] * self.num_rays)
        self.score = 0
        self.half_lap = False
        self.three_quarter_lap = False
        self.no_access_flag = False
        self.start_pos_x, self.start_pos_y = self.car_x, self.car_y
        self.ray_x = np.array([0] * self.num_rays)
        self.ray_y = np.array([0] * self.num_rays)

        # Return initial observation
        observation = np.array(self.ray_values.tolist() + [self.car_x, self.car_y, self.car_angle])
        return observation

    def render(self):
        # Main rendering loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        self.clock.tick(720)

        # Clear the screen
        self.screen.fill(self.white)

        # Draw circular path
        pygame.draw.circle(self.screen, self.brown, self.center, self.radius, self.path_thickness)
        
        # draw the inner circle within the circular track, where the car cannot go
        pygame.draw.circle(self.screen, self.black, self.center, self.no_path_radius, self.no_path_radius)
        
        # draw outer inaccessable circle
        pygame.draw.circle(self.screen, self.dark_green, self.center, self.no_path_outer_radius + (self.path_thickness), self.path_thickness)

        # Draw car as an arrow
        car_points = [
            (self.car_x + self.car_length * math.cos(self.car_angle), self.car_y - self.car_length * math.sin(self.car_angle)),
            (self.car_x + self.car_width * math.cos(self.car_angle - math.pi / 2), self.car_y - self.car_width * math.sin(self.car_angle - math.pi / 2)),
            (self.car_x - self.car_width * math.cos(self.car_angle), self.car_y + self.car_width * math.sin(self.car_angle)),
            (self.car_x + self.car_width * math.cos(self.car_angle + math.pi / 2), self.car_y - self.car_width * math.sin(self.car_angle + math.pi / 2))
        ]
        pygame.draw.polygon(self.screen, self.black, car_points)

        # Color the starting strip green by drawing a green horizontal rectangle along the starting strip
        pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(self.start_pos_x - self.path_thickness//2, self.center[1]-2, self.path_thickness, 4))
        
        # Color the half lap strip red by drawing a yellow horizontal rectangle along the half lap strip
        pygame.draw.rect(self.screen, (255, 255, 0), pygame.Rect(self.center[0] - self.radius, self.center[1]-2, self.path_thickness, 4))
        
        # Color the starting point blue by drawing a blue circle of radius 4
        pygame.draw.circle(self.screen, (0, 0, 255), (self.start_pos_x, self.start_pos_y), 4)
        
        # Plot points at the center of the circular path and another at 10 cm lower than the center of the circular path
        pygame.draw.circle(self.screen, (0, 0, 0), (self.center[0], self.center[1]), 4)
        pygame.draw.circle(self.screen, (255, 0, 0), (self.center[0], self.center[1] + 10), 4)
        pygame.draw.circle(self.screen, (0, 0, 255), (self.center[0], self.center[1] + self.radius - (self.path_thickness//2)), 4)

        # Draw radar rays
        for i in range(self.num_rays):
            # ray_x = self.car_x + self.ray_lengths[i] * math.cos(self.car_angle + self.ray_angles[i])
            # ray_y = self.car_y - self.ray_lengths[i] * math.sin(self.car_angle + self.ray_angles[i])
            pygame.draw.line(self.screen, self.black, (self.car_x, self.car_y), (self.ray_x[i], self.ray_y[i]), 1)

        # Draw score box
        score_box = pygame.Surface((self.window_size[0], 100))
        score_box.fill((200, 200, 200))
        font = pygame.font.Font(None, 24)
        text = font.render(f"State: {self.ray_values.tolist() + [self.car_x, self.car_y, self.car_angle]}  Score: {self.score}", True, (0, 0, 0))
        score_box.blit(text, (10, 10))
        self.screen.blit(score_box, (0, self.window_size[1] - 100))
        
        # Update the display
        pygame.display.flip()


# Initialize Pygame
# pygame.init()

# # Create environment instance
# env = ContinuousCarRadarEnv()

# # Example of using the environment
# state = env.reset()
# env.render()
# done_flag = False
# for i in range(5000):
#     if i%17 == 0:
#         action = 0
#     else:
#         action = 1
#     next_state, reward, done, _ = env.step(action)
#     if done:
#         done_flag = True
#         print("Goal reached successfully!")
    
#     env.render()
#     if done_flag:
#         break

# # Quit Pygame
# pygame.quit()
