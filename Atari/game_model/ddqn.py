#Internal imports
from game_model.base import BaseGameModel
from NeuralNet import NeuralNet
from utils.ReplayBuffer import ReplayBuffer

#External imports
import os
import h5py
import numpy as np

EXPLORATION_TEST = 0.02
EXPLORATION_STEPS = 850000
EXPLORATION_DECAY = (1.0-0.1)/EXPLORATION_STEPS

class DDQNGameModel(BaseGameModel):
	def __init__(self,game_name,input_dims,action_space,alpha=0.00025,data_paths=None):
		BaseGameModel.__init__(self,game_name,input_dims,action_space,data_paths=data_paths)

		self.local_save_path = os.path.join(self.model_path,'local-wts.h5')
		self.target_save_path = os.path.join(self.model_path,'target-wts.h5')

		self.local_model = NeuralNet(self.input_dims,self.action_space,learning_rate=alpha).model
		self.target_model = NeuralNet(self.input_dims,self.action_space,learning_rate=alpha).model

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
		DDQNGameModel.__init__(self,game_name,input_dims,action_space,alpha=alpha,data_paths=data_paths)

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
		self.memory = ReplayBuffer(max_size=self.replay_buffer_size,obs_shape=input_dims,data_paths=data_paths)

		self.params_save_path = os.path.join(self.model_path,'params.npz')

		self.drive = data_paths.drive

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

	def save_params(self):
		np.savez(self.params_save_path,epsilon=self.epsilon)
		print('Saved params...')

	def load_params(self):
		if os.path.isfile(self.params_save_path):
			with np.load(self.params_save_path) as data:
				self.epsilon = data['epsilon']
				print('Loaded params...')

	def act(self,obs):
		if np.random.rand() < self.epsilon or self.memory.meta_data['fill_size'] < self.replay_start_size:
			return self.action_space.sample()
		q_vals = self.local_model.predict(np.expand_dims(obs,axis=0),batch_size=1)
		return np.argmax(q_vals[0])

	def remember(self,curr_obs,action,reward,next_obs,done):
		self.memory.add(curr_obs,action,reward,next_obs,done)

	def step_update(self,tot_step):
		hist = None
		if self.memory.meta_data['fill_size'] < self.replay_start_size:
			return hist

		if tot_step % self.training_freq == 0:
			hist = self.replay()

		# self.update_epsilon()
		self.update_epsilon(tot_step)

		if tot_step % self.model_save_freq == 0:
			self.save_models()
			if self.drive:
				self.memory.save()
			
		if tot_step % self.target_network_update_freq == 0:
			self.reset_target_network()
		
		return hist

	def replay(self):
		if self.memory.meta_data['fill_size'] < self.batch_size:
			return
		curr_obs,action,reward,next_obs,done = self.memory.get_minibatch(self.batch_size)
		
		target = self.local_model.predict(curr_obs.astype(float)/255,batch_size=self.batch_size)

		done_mask = done.ravel()
		undone_mask = np.invert(done).ravel()

		target[done_mask,action[done_mask].ravel()] = reward[done_mask].ravel()

		Q_target = self.target_model.predict(next_obs.astype(float)/255,batch_size=self.batch_size)
		Q_future = np.max(Q_target[undone_mask],axis=1)

		target[undone_mask,action[undone_mask].ravel()] = reward[undone_mask].ravel() + self.gamma * Q_future

		fit = self.local_model.fit(curr_obs.astype(float)/255, target, batch_size=self.batch_size, verbose=0).history
		return fit

	def update_epsilon(self):
		self.epsilon = max(self.epsilon_min,self.epsilon - self.epsilon_decay)

	def update_epsilon1(self,t):
		tot_step_lim_fract = 1e6
		fract = min(float(t) / tot_step_lim_fract, 1.0)
		self.epsilon = max(self.epsilon_min,self.epsilon * fract)

	def reset_target_network(self):
		self.target_model.set_weights(self.local_model.get_weights())
		print('Target Network Reset...')

	'''
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

		fit = self.local_model.fit(update_input, update_target, batch_size=self.batch_size, verbose=0).history
		return fit
	'''
