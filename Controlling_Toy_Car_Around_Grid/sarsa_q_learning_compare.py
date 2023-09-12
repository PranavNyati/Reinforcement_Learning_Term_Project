import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import time
import numpy as np
import pygame
import pandas as pd

from my_env_control import Environment_Control

# read the csvs corresponding to the SARSA and Q-Learning algorithms containing the episode returns
sarsa_df = pd.read_csv("Sarsa(λ)_Episode_Returns_gamma_0.95_maze_16.csv")
q_learning_df = pd.read_csv("Q_Learning_Episode_Returns_gamma_0.95_maze_16.csv")

sarsa_returns = sarsa_df["Episode_Returns"].values
q_learning_returns = q_learning_df["Episode_Returns"].values

# calculate the average return for each episode for both SARSA and Q-Learning
sarsa_avg = []
ctr = 1
for i in range(len(sarsa_returns)):
    sarsa_avg.append(np.mean(sarsa_returns[:ctr]))
    ctr += 1
    
q_learning_avg = []
ctr = 1
for i in range(len(q_learning_returns)):
    q_learning_avg.append(np.mean(q_learning_returns[:ctr]))
    ctr += 1


num_episodes = len(sarsa_avg)

# plot the returns for each episode for both algorithms in the same plot
plt.figure(figsize=(10, 5))
plt.plot(sarsa_avg, label="SARSA(λ)")
plt.plot(q_learning_avg, label="Q-Learning")
plt.title("Comparison of SARSA(λ) and Q-Learning Performance over episodes")
plt.xlabel("Episode_Number")
plt.ylabel("Average Return per Episode")
plt.legend()
plt.grid(True)
plt.savefig("SARSA(λ)_vs_Q-Learning_Average_Return_gamma_0.95_maze_16.png")
