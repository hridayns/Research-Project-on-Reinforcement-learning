#Internal imports
from game_model.base import BaseGameModel
from NeuralNet import NeuralNet

#External imports
import os
import h5py
import pickle
import numpy as np
from sys import getsizeof
from random import sample
from collections import deque
from timeit import default_timer as timer

EXPLORATION_TEST = 0.02
EXPLORATION_STEPS = 850000
EXPLORATION_DECAY = (1.0-0.1)/EXPLORATION_STEPS

class DDQNGameModel(BaseGameModel):
	def __init__(self,game_name,input_dims,action_space,data_paths=None):
		BaseGameModel.__init__(self,game_name,input_dims,action_space,data_paths=data_paths)

		self.local_save_path = os.path.join(self.model_path,'local-wts.h5')
		self.target_save_path = os.path.join(self.model_path,'target-wts.h5')

		self.local_model = NeuralNet(self.input_dims,self.action_space).model
		self.target_model = NeuralNet(self.input_dims,self.action_space).model

		self.load_models()

	def load_models(self):
		if os.path.isfile(self.local_save_path):
			self.local_model.load_weights(self.local_save_path)
			print('Loaded Checkpoint: Local model...')

		if os.path.isfile(self.target_save_path):
			self.target_model.load_weights(self.target_save_path)
			print('Loaded Checkpoint: Target model...')

	def save_models(self):
		if os.path.isfile(self.local_save_path):
			os.remove(self.local_save_path)
		if os.path.isfile(self.target_save_path):
			os.remove(self.target_save_path)
		print('Checkpoint: Local and Target models saved...')

		self.local_model.save_weights(self.local_save_path)
		self.target_model.save_weights(self.target_save_path)


class DDQNPlayer(DDQNGameModel):
	def __init__(self,game_name,input_dims,action_space,data_paths=None):
		DDQNGameModel.__init__(self,game_name,input_dims,action_space,data_paths=data_paths)

		def act(self,obs):
			if np.random.rand() < EXPLORATION_TEST:
				return self.action_space.sample()
			q_vals = self.local_model.predict(obs,batch_size=1)
			return np.argmax(q_vals[0])

class DDQNLearner(DDQNGameModel):
	def __init__(self,game_name,input_dims,action_space,mem_size,gamma,batch_size,alpha,save_freq,target_train_freq,replay_start_size,train_freq,data_paths=None):
		DDQNGameModel.__init__(self,game_name,input_dims,action_space,data_paths=data_paths)

		self.reset_target_network()
		self.epsilon = 1.0
		self.epsilon_min = 0.1
		self.epsilon_decay = EXPLORATION_DECAY
		self.gamma = gamma
		self.target_network_update_freq = target_train_freq
		self.model_save_freq = save_freq
		self.batch_size = batch_size
		self.replay_start_size = replay_start_size
		self.training_freq = train_freq
		self.replay_buffer_size = mem_size
		self.memory = deque(maxlen=self.replay_buffer_size)

		self.params_save_path = os.path.join(self.model_path,'params.npz')
		self.replay_buffer_save_path = os.path.join(self.model_path,'replay-buffer.hdf5')

		self.drive = data_paths.drive
		if self.drive:
			self.load_replay_buffer()
		self.load_params()

		self.show_hyperparams()

	def show_hyperparams(self):
		print('Discount Factor (gamma): {}'.format(self.gamma))
		print('Batch Size: {}'.format(self.batch_size))
		print('Replay Buffer Size: {}'.format(self.replay_buffer_size))
		print('Training Frequency: {}'.format(self.training_freq))
		print('Model Save Frequency: {}'.format(self.model_save_freq))
		print('Target network update Frequency: {}'.format(self.target_network_update_freq))
		print('Replay start size: {}'.format(self.replay_start_size))

	def load_replay_buffer(self):
		self.show_saved_replay_buffer_size()
		start = timer()
		if os.path.isfile(self.replay_buffer_save_path):
			# with h5py.File(self.replay_buffer_save_path,'r') as f:
			# 	self.memory = deque(f['replay_buffer'])
			with open(self.replay_buffer_save_path,'rb') as handle:
				self.memory = pickle.load(handle)
				print('Replay Buffer loaded...')
		end = timer()
		print('Time taken: {} seconds'.format(end-start))

	def save_replay_buffer(self):
		self.show_saved_replay_buffer_size()
		start = timer()
		# with h5py.File(self.replay_buffer_save_path,'w') as f:
		# 	dset = f.create_dataset('replay_buffer',data=np.asarray(self.memory))
		with open(self.replay_buffer_save_path,'wb') as handle:
			pickle.dump(self.memory,handle,protocol=pickle.HIGHEST_PROTOCOL)
			print('Checkpoint replay buffer saved...')
		end = timer()
		print('Time taken: {} seconds'.format(end-start))

	def save_params(self):
		np.savez(self.params_save_path,epsilon=self.epsilon)
		print('Saved params...')

	def load_params(self):
		if os.path.isfile(self.params_save_path):
			with np.load(self.params_save_path) as data:
				self.epsilon = data['epsilon']
				print('Loaded params...')

	def show_saved_replay_buffer_size(self):
		mb_factor = (1024 * 1024)
		if os.path.isfile(self.replay_buffer_save_path):
			print('size of file: {} MB'.format(os.path.getsize(self.replay_buffer_save_path)/mb_factor))

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
			self.save_models()
			if self.drive:
				self.save_replay_buffer()

		if tot_step % self.target_network_update_freq == 0:
			self.reset_target_network()
		

	def replay(self):
		if len(self.memory) < self.batch_size:
			return
		batch = sample(self.memory,self.batch_size)

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

		fit = self.local_model.fit(update_input, update_target, batch_size=self.batch_size, verbose=0)

	def update_epsilon(self):
		self.epsilon = max(self.epsilon_min,self.epsilon - self.epsilon_decay)

	def reset_target_network(self):
		self.target_model.set_weights(self.local_model.get_weights())
		print('Target Network Reset...')