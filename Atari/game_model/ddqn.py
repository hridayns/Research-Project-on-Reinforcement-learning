#Internal imports
from game_model.base import BaseGameModel
from NeuralNet import NeuralNet

#External imports
import os
try:
	import cPickle as pickle
	print('Successfully imported cPickle...')
except:
	import pickle
	print('cPickle import failed. Loading Pickle instead...')
import numpy as np
# from shutil import copyfile
from sys import getsizeof
from random import sample
from collections import deque
from timeit import default_timer as timer

EXPLORATION_TEST = 0.02
EXPLORATION_STEPS = 850000
EXPLORATION_DECAY = (1.0-0.1)/EXPLORATION_STEPS

class DDQNGameModel(BaseGameModel):
	def __init__(self,game_name,input_dims,action_space,collab):
		BaseGameModel.__init__(self,game_name,input_dims,action_space)
		
		self.collab = collab
		if self.collab:
			self.model_path = self.collab_save_path

		self.local_save_path = os.path.join(self.model_path,'local-wts.h5')
		self.target_save_path = os.path.join(self.model_path,'target-wts.h5')

		self.local_model = NeuralNet(self.input_dims,self.action_space).model
		self.target_model = NeuralNet(self.input_dims,self.action_space).model

		if not os.path.exists(self.model_path):
			os.makedirs(self.model_path)

		if self.collab:
			print('Local Model save path: {}'.format(self.local_save_path))
			print('Target Model save path: {}'.format(self.target_save_path))

		self.load_checkpoint()

	def load_checkpoint(self):
		if os.path.isfile(self.local_save_path):
			self.local_model.load_weights(self.local_save_path)
			print('Loaded Checkpoint: Local model...')

		if os.path.isfile(self.target_save_path):
			self.target_model.load_weights(self.target_save_path)
			print('Loaded Checkpoint: Target model...')

	def save_checkpoint(self):
		if os.path.isfile(self.local_save_path):
			os.remove(self.local_save_path)
		if os.path.isfile(self.target_save_path):
			os.remove(self.target_save_path)
		print('Checkpoint: Local and Target models saved...')

		self.local_model.save_weights(self.local_save_path)
		self.target_model.save_weights(self.target_save_path)


class DDQNPlayer(DDQNGameModel):
	def __init__(self,game_name,input_dims,action_space):
		DDQNGameModel.__init__(self,game_name,input_dims,action_space)

		def act(self,obs):
			if np.random.rand() < EXPLORATION_TEST:
				return self.action_space.sample()
			q_vals = self.local_model.predict(obs,batch_size=1)
			return np.argmax(q_vals[0])

class DDQNLearner(DDQNGameModel):
	def __init__(self,game_name,input_dims,action_space,mem_size,gamma,batch_size,alpha,save_freq,target_train_freq,replay_start_size,train_freq,collab):
		DDQNGameModel.__init__(self,game_name,input_dims,action_space,collab)

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

		self.state_save_path = os.path.join(self.model_path,'state.npz')
		self.replay_buffer_save_path = os.path.join(self.model_path,'replay-buffer.pickle')
		self.score_buffer_save_path = os.path.join(self.model_path,'score-buffer.pickle')

		if self.collab:
			self.load_replay_buffer()

	def load_replay_buffer(self):
		self.show_saved_replay_buffer_size()
		start = timer()
		if os.path.isfile(self.replay_buffer_save_path):
			with open(self.replay_buffer_save_path, 'rb') as handle:
				self.memory = pickle.load(handle)
				print('Replay Buffer loaded...')
		end = timer()
		print('Time taken: {} seconds'.format(end-start))

	def show_hyperparams(self):
		print('Discount Factor (gamma): {}'.format(self.gamma))
		print('Batch Size: {}'.format(self.batch_size))
		print('Replay Buffer Size: {}'.format(self.replay_buffer_size))
		print('Training Frequency: {}'.format(self.training_freq))
		print('Model Save Frequency: {}'.format(self.model_save_freq))
		print('Target network update Frequency: {}'.format(self.target_network_update_freq))
		print('Replay start size: {}'.format(self.replay_start_size))

	def show_saved_replay_buffer_size(self):
		mb_factor = (1024 * 1024)
		if os.path.isfile(self.replay_buffer_save_path):
			print('size of file: {} MB'.format(os.path.getsize(self.replay_buffer_save_path)/mb_factor))

	def save_replay_buffer(self):
		# tmp = os.path.join(self.model_path,'replay-buffer-old.pickle')
		# if os.path.isfile(self.replay_buffer_save_path):
		# 	os.remove(self.replay_buffer_save_path)
			# copyfile(self.replay_buffer_save_path,tmp)
		self.show_saved_replay_buffer_size()
		start = timer()
		with open(self.replay_buffer_save_path, 'wb') as handle:
			pickle.dump(self.memory, handle, protocol=pickle.HIGHEST_PROTOCOL)
			print('Checkpoint replay buffer saved...')
		# if os.path.isfile(tmp):
			# os.remove(tmp)
		end = timer()
		print('Time taken: {} seconds'.format(end-start))

	'''
	def calc_buffer_size(self):
		tot = 0
		for data in self.memory[0]:
			if type(data) is np.ndarray:
				tot += data.nbytes
			else:
				tot += getsizeof(data)
		block_size = tot / (1024 * 1024)

		print('Block size: {}'.format(block_size))
		for i in [100,1000,5000,10000,25000,50000,100000,1000000]:
			print('Buffer size of {} blocks: {} MB'.format(i,i*block_size))
	'''

	def act(self,obs):
		if np.random.rand() < self.epsilon or len(self.memory) < self.replay_start_size:
			return self.action_space.sample()
		q_vals = self.local_model.predict(obs,batch_size=1)
		return np.argmax(q_vals[0])

	def remember(self,curr_obs,action,reward,next_obs,done):
		self.memory.append([curr_obs,action,reward,next_obs,done])
		
		'''
		mem = deque(maxlen=10000)
		for i in range(10000):
			curr_obs_data = np.full(shape=curr_obs.shape,fill_value=255,dtype=np.uint8)
			next_obs_data = np.full(shape=next_obs.shape,fill_value=255,dtype=np.uint8)
			action_data = np.uint8(1)
			reward_data = np.int8(-1)
			done_data = False
			mem.append([curr_obs_data,action_data,reward_data,next_obs_data,done_data])
		
		pickled_mem = pickle.dumps(mem)
		print('Size of pickled mem: {}'.format(getsizeof(pickled_mem)))
		input()
		'''
		'''
		block = [curr_obs,action,reward,next_obs,done]
		pickle_block = pickle.dumps(block)
		print('Pickled block size: {}'.format(getsizeof(pickle_block)))
		start = timer()

		tot_factor = self.replay_buffer_size/(1024*1024)
		curr_obs_size_b = curr_obs.nbytes
		action_size_b = getsizeof(action)
		reward_size_b = getsizeof(reward)
		next_obs_size_b = next_obs.nbytes
		done_size_b = getsizeof(done)

		mb_factor = (1024*1024)
		curr_obs_size = curr_obs_size_b/mb_factor
		next_obs_size = next_obs_size_b/mb_factor
		action_size = action_size_b/mb_factor
		reward_size = reward_size_b/mb_factor
		done_size = done_size_b/mb_factor

		block_size_b = curr_obs_size_b + action_size_b + reward_size_b + next_obs_size_b + done_size_b
		block_size = block_size_b/mb_factor

		tot_size_b = block_size_b * self.replay_buffer_size
		tot_size = block_size * self.replay_buffer_size

		print('Bytes analysis: ')
		print('Total Size: {} bytes'.format(tot_size_b))
		print('Block size: {} bytes'.format(block_size_b))
		print('Curr Obs Size: {} bytes'.format(curr_obs_size_b))
		print('Action Size: {} bytes'.format(action_size_b))
		print('Reward Size: {} bytes'.format(reward_size_b))
		print('Next Obs Size: {} bytes'.format(next_obs_size_b))
		print('Done Size: {} bytes'.format(done_size_b))

		print('MB analysis: ')
		print('Total Size: {} MB'.format(tot_size))
		print('Block size: {} MB'.format(block_size))
		print('Curr Obs Size: {} MB'.format(curr_obs_size))
		print('Action Size: {} MB'.format(action_size))
		print('Reward Size: {} MB'.format(reward_size))
		print('Next Obs Size: {} MB'.format(next_obs_size))
		print('Done Size: {} MB'.format(done_size))
		
		end = timer()
		print('Time elapsed: {}'.format(end-start))
		input()
		'''

	def step_update(self,tot_step):
		if len(self.memory) < self.replay_start_size:
			return
		'''
		if len(self.memory) == self.replay_buffer_size:
			# self.calc_buffer_size()
			pickle_mem = pickle.dumps(self.memory)
			print(getsizeof(pickle_mem))
			input()
		'''
		if tot_step % self.training_freq == 0:
			self.replay()

		self.update_epsilon()

		if tot_step % self.model_save_freq == 0:
			self.save_checkpoint()
			if self.collab:
				# print('saving replay buffer...')
				self.save_replay_buffer()

		if tot_step % self.target_network_update_freq == 0:
			self.reset_target_network()

	def replay(self):
		batch = np.asarray(sample(self.memory,self.batch_size))
		if len(batch) < self.batch_size:
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

		fit = self.local_model.fit(update_input, update_target, batch_size=self.batch_size, verbose=0)

	def update_epsilon(self):
		self.epsilon = max(self.epsilon_min,self.epsilon - self.epsilon_decay)

	def reset_target_network(self):
		self.target_model.set_weights(self.local_model.get_weights())
