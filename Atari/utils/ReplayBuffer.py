import os
import psutil
import h5py
import numpy as np
from timeit import default_timer as timer

class ReplayBuffer:
	def __init__(self,max_size=140000,obs_shape=(84,84,4),data_paths=None):
		self.max_size = max_size
		self.buffer_ptr = 0
		self.fill_size = 0
		self.replay_buffer_save_path = os.path.join(data_paths.save_folders['model'],'replay-buffer.hdf5')
		self.buffer_metadata_save_path = os.path.join(data_paths.save_folders['model'],'buffer-metadata.npz')

		self.curr_obs = np.empty(shape=(self.max_size,obs_shape[0],obs_shape[1],obs_shape[2]),dtype=np.uint8)
		self.action = np.empty(shape=(self.max_size,1),dtype=np.uint8)
		self.reward = np.empty(shape=(self.max_size,1),dtype=np.int8)
		self.next_obs = np.empty(shape=(self.max_size,obs_shape[0],obs_shape[1],obs_shape[2]),dtype=np.uint8)
		self.done = np.empty(shape=(self.max_size,1),dtype=np.bool)
		
		if data_paths.drive:
			self.load()

		self.load()

	def add(self,curr_obs,action,reward,next_obs,done):
		self.curr_obs[self.buffer_ptr] = curr_obs
		self.action[self.buffer_ptr] = action
		self.reward[self.buffer_ptr] = reward
		self.next_obs[self.buffer_ptr] = next_obs
		self.done[self.buffer_ptr] = done
		self.buffer_ptr += 1
		self.fill_size = max(self.fill_size,self.buffer_ptr)
		self.buffer_ptr = self.buffer_ptr % self.max_size

	def get_minibatch(self,batch_size=32):
		sample_idx = np.random.choice(self.fill_size,batch_size,replace=False)
		curr_obs_batch = self.curr_obs[sample_idx,...]
		action_batch = self.action[sample_idx,...]
		reward_batch = self.reward[sample_idx,...]
		next_obs_batch = self.next_obs[sample_idx,...]
		done_batch = self.done[sample_idx,...]

		return curr_obs_batch,action_batch,reward_batch,next_obs_batch,done_batch

	def save(self):
		print('Saving replay buffer...')
		self.show_saved_replay_buffer_size()
		self.show_RAM_usage()
		start = timer()
		with h5py.File(self.replay_buffer_save_path,'w') as f:
			curr_obs = f.create_dataset('curr_obs',data=self.curr_obs)
			action = f.create_dataset('action',data=self.action)
			reward = f.create_dataset('reward',data=self.reward)
			next_obs = f.create_dataset('next_obs',data=self.next_obs)
			done = f.create_dataset('done',data=self.done)
			print('Replay buffer saved...')
		end = timer()
		print('Time taken: {} seconds'.format(end-start))
		print('Saving buffer metadata...')
		np.savez(self.buffer_metadata_save_path,buffer_ptr=self.buffer_ptr,fill_size=self.fill_size)
		print('Buffer metadata saved...')

	def load(self):
		self.show_saved_replay_buffer_size()
		self.show_RAM_usage()
		start = timer()
		if os.path.isfile(self.replay_buffer_save_path):
			print('Loading replay buffer...')
			with h5py.File(self.replay_buffer_save_path,'r') as f:
				self.curr_obs = np.asarray(f['curr_obs'])
				self.action = np.asarray(f['action'])
				self.reward = np.asarray(f['reward'])
				self.next_obs = np.asarray(f['next_obs'])
				self.done = np.asarray(f['done'])
				print('Replay Buffer Loaded...')
			end = timer()
			print('Time taken: {} seconds'.format(end-start))

		if os.path.isfile(self.buffer_metadata_save_path):
			print('Loading buffer metadata...')
			with np.load(self.buffer_metadata_save_path) as data:
				self.buffer_ptr = data['buffer_ptr']
				self.fill_size = data['fill_size']
				print('Buffer metadata Loaded...')

	def show_saved_replay_buffer_size(self):
		mb_factor = (1024 * 1024)
		if os.path.isfile(self.replay_buffer_save_path):
			print('size of file: {} MB'.format(os.path.getsize(self.replay_buffer_save_path)/mb_factor))

	def show_RAM_usage(self):
		py = psutil.Process(os.getpid())
		print('RAM usage: {} GB'.format(py.memory_info()[0]/2. ** 30))

	def show_replay_buffer(self):
		print(self.curr_obs.shape)
		print(self.next_obs.shape)
		print(self.action.shape)
		print(self.reward.shape)
		print(self.done.shape)
		input()

