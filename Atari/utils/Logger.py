import os
import numpy as np
from collections import deque
import h5py


class Logger:
	def __init__(self,save_dirs,log_types=['avg_score','high_score','low_score','avg_loss','avg_acc'],log_freq=10,mode='train'):
		self.log_types = log_types
		self.log_freq = log_freq
		self.data = {
			'episode' : 0,
			'timesteps' : 0,
			'ep_steps': 0,
			'score': 0,
			'avg_score' : 0,
			'avg_score_100': 0,
			'best_avg_score_100': -1000,
			'high_score' : -1000,
			'low_score' : 1000,
			'loss': 0,
			'avg_loss': 0,
			'acc': 0,
			'avg_acc': 0
		}

		self.score_window = deque(maxlen=100)

		self.state_save_path = os.path.join(save_dirs['checkpoints'],'state.npz')
		self.score_window_save_path = os.path.join(save_dirs['checkpoints'],'score_window.hdf5')

		if mode == 'train':
			self.load_state()

	def update_best_score(self):
		if self.data['best_avg_score_100'] is 0 or self.data['avg_score_100'] > self.data['best_avg_score_100']:
			if self.log_freq is not None:
				print('Saving model due to mean reward increase: {} -> {}'.format(self.data['best_avg_score_100'],self.data['avg_score_100']))
				self.data['best_avg_score_100'] = self.data['avg_score_100']
				return True

	def save_state(self):
		np.savez(self.state_save_path,**self.data)
		print('State Saved...')

		with h5py.File(self.score_window_save_path,'w') as f:
			dset = f.create_dataset('score_window',data=np.array(self.score_window))

		print('Score Window Saved...')

	def load_state(self):
		if os.path.isfile(self.state_save_path):
			with np.load(self.state_save_path) as state:
				for k in self.data:
					self.data[k] = np.asscalar(state[k])
			
			print('State Loaded...')

		if os.path.isfile(self.score_window_save_path):
			with h5py.File(self.score_window_save_path,'r') as f:
				self.score_window = deque(f['score_window'])
			
			print('Score Window Loaded...')

		print('Resuming from...')
		self.show_state()

	def update_state(self,ep_steps,score,loss,acc):
		self.data['episode'] += 1
		self.data['timesteps'] += ep_steps
		self.data['ep_steps'] = ep_steps

		self.data['score'] = score
		self.data['loss'] = loss
		self.data['acc'] = acc

		self.score_window.append(score)
		self.data['avg_score_100'] = np.round(np.mean(self.score_window),1)
		# if self.data['avg_score_100'] > self.data['best_avg_score_100']:
		# 	self.data['best_avg_score_100'] = self.data['avg_score_100']

		self.data['avg_loss'] = (self.data['avg_loss'] * (self.data['episode']-1) + self.data['loss']) / self.data['episode']
		self.data['avg_acc'] = (self.data['avg_acc'] * (self.data['episode']-1) + self.data['acc']) / self.data['episode']
		self.data['avg_score'] = (self.data['avg_score'] * (self.data['episode']-1) + self.data['score']) / self.data['episode']
		self.data['high_score'] = max(self.data['score'],self.data['high_score'])
		self.data['low_score'] = min(self.data['score'],self.data['low_score'])

	def show_state(self):
		print('{:-^50}'.format('Episode ' + str(self.data['episode'])))
		print('Avg Score for last 100 episodes: {}'.format(self.data['avg_score_100']))
		# print('Score: {}'.format(self.data['score']))
		print('Global Timesteps: {}'.format(self.data['timesteps']))
		# print('Timesteps per episode: {}'.format(self.data['ep_steps']))
		# print('Loss: {}'.format(self.data['loss']))
		# print('Accuracy: {}'.format(self.data['acc']))
		if 'avg_score' in self.log_types:
			print('Avg Score: {}'.format(self.data['avg_score']))
		if 'high_score' in self.log_types:
			print('Highest Score: {}'.format(self.data['high_score']))
		if 'low_score' in self.log_types:
			print('Lowest Score: {}'.format(self.data['low_score']))
		if 'avg_loss' in self.log_types:
			print('Avg Loss: {}'.format(self.data['avg_loss']))
		if 'avg_acc' in self.log_types:
			print('Avg Accuracy: {}'.format(self.data['avg_acc']))
		print('{:-^50}'.format(''))

	def log_state(self,done,exploration_perc):
		if done and self.log_freq is not None and self.data['episode'] % self.log_freq == 0: 
			print('% time spent exploring: {}'.format(int(100 * exploration_perc)))
			self.show_state()