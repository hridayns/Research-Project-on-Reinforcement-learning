#Internal imports
from game_model.base import BaseGameModel
from NeuralNet import NeuralNet

#External imports
import numpy as np
import os
from random import sample
from collections import deque


GAMMA = 0.99
MEMORY_SIZE = 50000#900000
BATCH_SIZE = 32
TRAINING_FREQUENCY = 4
TARGET_NETWORK_UPDATE_FREQUENCY = 40000
MODEL_PERSISTENCE_UPDATE_FREQUENCY = 10000
REPLAY_START_SIZE = 50000

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_TEST = 0.02
EXPLORATION_STEPS = 850000
EXPLORATION_DECAY = (EXPLORATION_MAX-EXPLORATION_MIN)/EXPLORATION_STEPS

# EPISODE = 0
# GSTEP = 0

class DDQNGameModel(BaseGameModel):
	def __init__(self,game_name,input_dims,action_space):
		BaseGameModel.__init__(self,game_name,input_dims,action_space)
		
		self.local_save_path = os.path.join(self.model_path,'local-wts.h5')
		self.target_save_path = os.path.join(self.model_path,'target-wts.h5')
		self.chkpt_save_path = os.path.join(self.model_path,'chkpt.npz')

		self.local_model = NeuralNet(self.input_dims,self.action_space).model
		self.target_model = NeuralNet(self.input_dims,self.action_space).model

		# self.epsilon = EXPLORATION_MAX
		# self.ep_init = EPISODE
		# self.g_step = GSTEP

		if os.path.isfile(self.local_save_path):
			self.local_model.load_weights(self.local_save_path)
		if os.path.isfile(self.target_save_path):
			self.target_model.load_weights(self.target_save_path)
		# if os.path.isfile(self.chkpt_save_path):
		# 	data = np.load(self.chkpt_save_path)
		# 	self.epsilon = data['epsilon']
		# 	self.ep_init = data['episode']
		# 	self.g_step = data['g_step']


		def save_checkpoint(self):
			if os.path.isfile(self.local_save_path):
				os.remove(self.local_save_path)
			if os.path.isfile(self.target_save_path):
				os.remove(self.target_save_path)
			# if os.path.isfile(self.chkpt_save_path):
			# 	os.remove(self.chkpt_save_path)

			self.local_model.save_weights(self.local_save_path)
			self.target_model.save_weights(self.target_save_path)
			# np.savez()


class DDQNPlayer(DDQNGameModel):
	def __init__(self,game_name,input_dims,action_space):
		DDQNGameModel.__init__(self,game_name,input_dims,action_space)

		def act(self,obs):
			if np.random.rand() < EXPLORATION_TEST:
				return self.action_space.sample()
			q_vals = self.local_model.predict(obs,batch_size=1)
			return np.argmax(q_vals[0])

class DDQNLearner(DDQNGameModel):
	def __init__(self,game_name,input_dims,action_space):
		DDQNGameModel.__init__(self,game_name,input_dims,action_space)

		self.reset_target_network()
		self.epsilon = EXPLORATION_MAX
		self.epsilon_min = EXPLORATION_MIN
		self.epsilon_decay = EXPLORATION_DECAY
		self.gamma = GAMMA
		self.target_network_update_freq = TARGET_NETWORK_UPDATE_FREQUENCY
		self.model_save_freq = MODEL_PERSISTENCE_UPDATE_FREQUENCY
		self.batch_size = BATCH_SIZE
		self.replay_start_size = REPLAY_START_SIZE
		self.training_freq = TRAINING_FREQUENCY
		self.memory = deque(maxlen=MEMORY_SIZE)


	def act(self,obs):
		if np.random.rand() < self.epsilon or len(self.memory) < self.replay_start_size:
			return self.action_space.sample()
		q_vals = self.local_model.predict(obs,batch_size=1)
		return np.argmax(q_vals[0])

	def remember(self,curr_obs,action,reward,next_obs,done):
		self.memory.append([curr_obs,action,reward,next_obs,done])

	def step_update(self,tot_step):
		if len(self.memory) < self.replay_start_size:
			return
		if tot_step % self.training_freq == 0:
			self.replay()

		self.update_epsilon()

		if tot_step % self.model_save_freq == 0:
			self.save_checkpoint()

		if tot_step % self.target_network_update_freq == 0:
			self.reset_target_network()

	def replay(self):
		batch = np.asarray(sample(self.memory,self.batch_size))
		if len(batch) < BATCH_SIZE:
			return

		update_input = np.zeros((self.batch_size,self.input_dims[0],self.input_dims[1],self.input_dims[2]))
		update_target = np.zeros((self.batch_size,self.action_space.n))

		for i in range(self.batch_size):
			curr_obs, action, reward, next_obs, done = batch[i]
			target = self.local_model.predict(curr_obs)

			if done:
				target[0][action] = reward
			else:
				Q_future = np.max(self.target_model.predict(next_obs)[0])
				target[0][action] = reward + self.gamma * Q_future

			update_input[i] = curr_obs
			update_target[i] = target

	def update_epsilon(self):
		self.epsilon = max(EXPLORATION_MIN,self.epsilon - self.epsilon_decay)

	def reset_target_network(self):
		self.target_model.set_weights(self.local_model.get_weights())
