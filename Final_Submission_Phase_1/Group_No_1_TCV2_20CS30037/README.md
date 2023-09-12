# REINFORCEMENT LEARNING TERM PROJECT (PRANAV NYATI: 20CS30037)
## PHASE 1 : CONTROLLING TOY CAR AROUND GRID

## Requirements:
- Python 3.8
- pygame
- matplotlib
- numpy
- pandas

## Instructions to run the codes:
1. Install pygame using the following command:
```
pip install pygame
```
2. To run the code for Value Iteration, set the grid-size, max_no of iterations, gamma and precision in the value_iteration.py file and run the following command:
```
python3 value_iteration.py
```
3. To run the code for SARSA-Lambda, set the grid-size, no. of episodes, gamma, epsilon, alpha and lambda in the sarsa_lambda.py file and run the following command:
```
python3 sarsa_lambda.py
```
4. To run the code for Q-Learning, set the grid-size, no. of episodes, gamma, epsilon and alpha in the q_learning.py file and run the following command:
```
python3 q_learning.py
```
5. To compare the results of SARSA-Lambda and Q-Learning, run the following command:
```
python3 sarsa_q_learning_compare.py
```
6. Results for different experiments are in the Results_and_Plots folder as follows:
- Value Iteration: Results_and_Plots/VI_forward_reward_goal_2x_reward_maze_16:

  Contains the analysis for varying gamma for maze size 16
- SARSA-Lambda: Results_and_Plots/Sarsa_lambda_forward_reward_goal_2x_reward_gamma_0.95_maze_16: 
  
  Contains the analysis for varying number of episodes with and without epsilon decay for maze size 16 and gamma 0.95
- Q-Learning: Results_and_Plots/Q_l_forward_reward_goal_2x_reward_gamma_0.95_maze_16: 
  
  Contains the analysis for varying number of episodes with and without epsilon decay for maze size 16 and gamma 0.95
- Comparison of SARSA-Lambda and Q-Learning: Results_and_Plots/Sarsa_Q_Learning_Comparison: 
  
  Contains the difference in learning curves for SARSA-Lambda and Q-Learning for maze size 16 and gamma 0.95 for different number of episodes