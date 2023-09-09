import numpy as np
import matplotlib.pyplot as plt
import copy

from myenv import Environment

class Temporal_difference:
    def __init__(self, mdp, gamma, n_steps):
        self.mdp = mdp
        self.gamma = gamma
        self.n_steps = n_steps
        self.final_optimal_value_function = None
        self.final_optimal_policy = None
        
    def temporal_difference(self, num_episodes = 1000, )