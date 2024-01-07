# REINFORCEMENT LEARNING TERM PROJECT (PRANAV NYATI: 20CS30037)
## PHASE 2 : CONTROLLING TOY CAR AROUND CONTINUOUS CIRCULAR TRACK USING DEEP REINFORCEMENT LEARNING

## Requirements:
- Python 3.8
- pygame
- matplotlib
- numpy
- pandas
- Pytorch (Version 1.12.0)

## Instructions to run the codes:
1. Install pygame using the following command:
```
pip install pygame
```
2. To train an agent for a particular algorithm, uncomment the section of code having the comment "# INITIALIZE THE AGENT AND TRAIN IT", in the main function of the respective algorithm's file. Similarly, to test a trained agent, uncomment the section of code having the comment "# TEST THE TRAINED AGENT", in the main function of the respective algorithm's file.

3. To run the code for DDQN, set the desired configs in Config class of ddqn.py file and run the following command:
```
python3 ddqn.py
```
4. To run the code for REINFORCE, set the desired configs in Config class of reinforce.py file and run the following command:
```
python3 reinforce.py
```
5. To run the code for A3C, set the desired configs in Config class of a3c.py file and run the following command:
```
python3 a3c.py
```
6. To run the code for GAE, set the desired configs in Config class of gae.py file and run the following command:
```
python3 gae.py
```
7. Results of training of above algorithms (Results include saved model weights and learning curves) can be found in the respective train_logs folder.