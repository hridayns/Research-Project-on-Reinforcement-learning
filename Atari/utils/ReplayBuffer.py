import os
import psutil
import h5py
import numpy as np
from timeit import default_timer as timer

class ReplayBuffer:
	def __init__(self,save_dirs,buffer_size=10000,obs_shape=(84,84,4)):
		self.buffer_size = buffer_size
		self.replay_buffer_save_path = os.path.join(save_dirs['checkpoints'],'replay-buffer.hdf5')

		self.data = {
			'curr_obs': np.empty(shape=(self.buffer_size,obs_shape[0],obs_shape[1],obs_shape[2]),dtype=np.uint8),
			'action': np.empty(shape=(self.buffer_size,1),dtype=np.uint8),
			'reward': np.empty(shape=(self.buffer_size,1),dtype=np.int8),
			'next_obs': np.empty(shape=(self.buffer_size,obs_shape[0],obs_shape[1],obs_shape[2]),dtype=np.uint8),
			'done': np.empty(shape=(self.buffer_size,1),dtype=np.bool)
		}

		self.meta_data = {
			'buffer_ptr': 0,
			'fill_size': 0
		}

	def add(self,curr_obs,action,reward,next_obs,done):
		idx = self.meta_data['buffer_ptr']
		fill_size = self.meta_data['fill_size']

		self.data['curr_obs'][idx] = curr_obs
		self.data['action'][idx] = action
		self.data['reward'][idx] = reward
		self.data['next_obs'][idx] = next_obs
		self.data['done'][idx] = done

		idx += 1
		fill_size = max(fill_size,idx)
		idx = idx % self.buffer_size

		self.meta_data['buffer_ptr'] = idx
		self.meta_data['fill_size'] = fill_size

	def get_minibatch(self,batch_size=32):
		sample_idx = np.random.choice(self.meta_data['fill_size'],batch_size,replace=False)
		curr_obs_batch = self.data['curr_obs'][sample_idx,...]
		action_batch = self.data['action'][sample_idx,...]
		reward_batch = self.data['reward'][sample_idx,...]
		next_obs_batch = self.data['next_obs'][sample_idx,...]
		done_batch = self.data['done'][sample_idx,...]

		return curr_obs_batch,action_batch,reward_batch,next_obs_batch,done_batch

	def save(self):
		print('Saving Replay Buffer...')
		start = timer()
		with h5py.File(self.replay_buffer_save_path,'w') as f:
			for i in self.meta_data.keys():
				f.attrs[i] = self.meta_data[i]
			for k in self.data.keys():
				f.create_dataset(k,data=self.data[k],compression='gzip')

		print('Replay Buffer Saved...')
		end = timer()
		print('Time taken: {} seconds'.format(end-start))

	def load(self):
		if os.path.isfile(self.replay_buffer_save_path):
			print('Loading Replay Buffer...')
			start = timer()
			with h5py.File(self.replay_buffer_save_path,'r') as f:
				for i in f.attrs.keys():
					self.meta_data[i] = f.attrs[i]
				for k in list(f.keys()):
					self.data[k] = np.array(f[k])
			print('Replay Buffer Loaded...')
			end = timer()
			print('Time taken: {} seconds'.format(end-start))
		else:
			print('No existing Replay Buffer found...')


	def show_saved_replay_buffer_size(self):
		mb_factor = (1024 * 1024)
		if os.path.isfile(self.replay_buffer_save_path):
			print('size of file: {} MB'.format(os.path.getsize(self.replay_buffer_save_path)/mb_factor))

	def show_RAM_usage(self):
		py = psutil.Process(os.getpid())
		print('RAM usage: {} GB'.format(py.memory_info()[0]/2. ** 30))

	'''
	def show_replay_buffer(self):
		for k in self.data.keys():
			print('{}: {}'.format(k,self.data[k].shape))
	'''