# Classic Control

This directory contains problems from the "Classic Control" section of the OpenAI Gym [environments](https://gym.openai.com/envs/#classic_control).

## Files

 - DQN-CartPole
 - DQN-MountainCar

## DQN-CartPole

Contains code for a DQN written from scratch using Python and NumPy. It implements a DDQN (Double DQN) with experience replay, where the "Model" network is used to predict optimal actions from the observations of the environment, and a "Target" network is trained every few episodes to help in predicting the true Q-values of actions taken by the agent. This is not really required for the simple CartPole problem, but demonstrates the power of a DDQN, as it solves the environment in only about 20 episodes.

## DQN-MountainCar  

Contains code for a DQN written from scratch using Python and NumPy. It implements a DDQN (Double DQN) with experience replay, where the "Model" network is used to predict optimal actions from the observations of the environment, and a "Target" network is trained every few episodes to help in predicting the true Q-values of actions taken by the agent. This is required for the MountainCar problem, since there is no real "reward" that the agent gets unless it reaches the goal. Using the same "model" network as its target would cause the neural network to chase a moving target, and never converge to a solution. The current implementation solves the environment in about 5000 episodes.

## Running

To run, follow the steps:

 1. Train the agent: `python *filename*`
 Example: `python DQN-CartPole.py`
 2. Test the agent: `python *filename* test`
 Example: `python DQN-CartPole.py test`

## Further changes

Feel free to make changes to the hyperparameters to see how it affects the agent. Or just use it to understand Q-networks.
