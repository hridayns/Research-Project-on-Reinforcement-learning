import gym
import numpy as np
import random
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import load_model
import sys
import os
import pickle

from pathlib import Path
from collections import deque


np.random.seed(42)
ENV_NAME = 'MountainCar-v0'
SAVE_FOLDER = os.path.join(os.getcwd(),'model-saves')
if not os.path.exists(SAVE_FOLDER):
    os.mkdir(SAVE_FOLDER)
LOCAL_WEIGHTS_SAVE = os.path.join(SAVE_FOLDER,ENV_NAME + '-DQN-local-weights.h5')
TARGET_WEIGHTS_SAVE = os.path.join(SAVE_FOLDER,ENV_NAME + '-DQN-target-weights.h5')
TRAIN_CHKPT_SAVE = os.path.join(SAVE_FOLDER,ENV_NAME + '-DQN-chkpt.npz')
REPLAY_BUFFER_SAVE = os.path.join(SAVE_FOLDER,ENV_NAME + '-DQN-replay-buffer.pickle')

EPISODES = 2000
render = False
training = True
if(len(sys.argv) > 1):
	if sys.argv[1].lower() == 'test':
		training = False

class DQN:
    def __init__(self, env):
        self.env     = env
        self.memory  = deque(maxlen=100000)
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.99
        self.learning_rate = 0.0001
        self.batch_size = 64

        self.model        = self.create_model()
        self.target_model = self.create_model()

        self.load_model()

    def create_model(self):
        model   = Sequential()
        state_shape  = self.env.observation_space.shape
        model.add(Dense(256, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def perform(self,state):
        return np.argmax(self.model.predict(state)[0])

    def remember(self,curr_obs,action,reward,next_obs,done):
    	self.memory.append([curr_obs,action,reward,next_obs,done])

    def replay(self):
        if len(self.memory) < self.batch_size: 
            return

        samples = random.sample(self.memory, self.batch_size)
        for sample in samples:
            curr_obs, action, reward, next_obs, done = sample
            target = self.model.predict(curr_obs)
            if done:
                target[0][action] = reward
            else:   
                Q_future = max(self.target_model.predict(next_obs)[0])
                target[0][action] = reward + self.gamma * Q_future
            self.model.fit(curr_obs, target, epochs=1, verbose=0)

    def target_train(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self):
        self.model.save(LOCAL_WEIGHTS_SAVE)
        self.target_model.save(TARGET_WEIGHTS_SAVE)
        with open(REPLAY_BUFFER_SAVE, 'wb') as handle:
            pickle.dump(self.memory, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self):
        if Path(LOCAL_WEIGHTS_SAVE).exists():
            self.model = load_model(LOCAL_WEIGHTS_SAVE)
        if Path(TARGET_WEIGHTS_SAVE).exists():
            self.target_model = load_model(TARGET_WEIGHTS_SAVE)
        if Path(REPLAY_BUFFER_SAVE).exists():
            with open(REPLAY_BUFFER_SAVE, 'rb') as handle:
                self.memory = pickle.load(handle)

def reshape_input(X):
	X = X.reshape(-1,X.shape[0])
	return X

def train():
    env = gym.make(ENV_NAME)
    target_reward = -110
    reward_window = deque(maxlen=100)
    consolidation_counter = 0

    dqn_agent = DQN(env=env)
    ep_start = 0
    if Path(TRAIN_CHKPT_SAVE).exists():
        train_chkpt = np.load(TRAIN_CHKPT_SAVE)
        ep_start = train_chkpt['ep']
        dqn_agent.epsilon = train_chkpt['epsi']

    for ep in range(ep_start,EPISODES):
        curr_obs = env.reset()
        curr_obs = reshape_input(curr_obs)
        total_r = 0
        step = 1
        while True:
            # if ep % 5 == 0:
            #     env.render()
            action = dqn_agent.act(curr_obs)
            next_obs, reward, done, _ = env.step(action)
            next_obs = reshape_input(next_obs)

            total_r += reward
            dqn_agent.remember(curr_obs,action,reward,next_obs,done)
            curr_obs = next_obs
            
            dqn_agent.replay()
            step += 1
            if done:
                # env.close()
                dqn_agent.target_train()
                break
        np.savez(TRAIN_CHKPT_SAVE,ep=ep,epsi=dqn_agent.epsilon)
        reward_window.append(total_r)
        dqn_agent.update_epsilon()
        print('Episode {} : Reward = {}'.format(ep,total_r))
        if ep % 50 == 0:
            dqn_agent.save_model()
        avg_reward_window = np.mean(reward_window)
        if avg_reward_window >= target_reward:
            consolidation_counter += 1
            if consolidation_counter >= 5:
                print('Completed training with avg reward {} over last 100 episodes. Training ran for a total of {} episodes.'.format(avg_reward_window,ep+1))
                input()
                return
            else:
                consolidation_counter = 0
    
    print('INCOMPLETE training with avg reward {} over last 100 episodes. Training ran for a total of {} episodes.'.format(avg_reward_window,ep+1))
    return
    
def test():
    env = gym.make(env_name)
    trained_agent = DQN(env=env)
    trained_agent.load_model()
    curr_obs = env.reset()
    curr_obs = reshape_input(curr_obs)
    steps = 0
    while True:
        env.render()
        action = trained_agent.perform(curr_obs)
        next_obs, reward, done, _ = env.step(action)
        next_obs = reshape_input(next_obs)
        curr_obs = next_obs
        steps += 1
        if done:
            break
    print('Completed in ',steps,' steps')

if __name__ == "__main__":
	if training:
		train()
	else:
		test()