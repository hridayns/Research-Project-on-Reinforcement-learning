import os
import numpy as np
import pickle


class Logger:
	def __init__(self,env_name,log_types=['avg_score','highest_score','lowest_score'],save_interval=10,data_paths=None):
		self.env_name = env_name
		self.log_types = log_types
		self.save_interval = save_interval

		self.state_save_path = os.path.join(data_paths.root_path,'state.npz')

		self.log_data = {
			'epoch' : 0,
			'ts' : 0,
			't': 0,
			'score': 0,
			'avg_score' : 0,
			'high_score' : -1000,
			'low_score' : 1000
		}


		self.load_state()

	# def print_types(self):
	# 	for k in self.log_data:
	# 		print(type(self.log_data[k]))
	# 	print(self.log_data)
	# 	input()

	def save_state(self):
		np.savez(
			self.state_save_path,
			epoch=self.log_data['epoch'],
			ts=self.log_data['ts'],
			t=self.log_data['t'],
			score=self.log_data['score'],
			avg_score=self.log_data['avg_score'],
			high_score=self.log_data['high_score'],
			low_score=self.log_data['low_score']
		)
		print('State Saved...')

	def load_state(self):
		if os.path.isfile(self.state_save_path):
			with np.load(self.state_save_path) as state:
				for k in self.log_data:
					self.log_data[k] = np.asscalar(state[k])

			print('State Loaded...')
			print('Resuming from...')
			self.show_state()

	def show_state(self):
		print('{:-^50}'.format('Episode ' + str(self.log_data['epoch'])))
		print('Global Timesteps: {}'.format(self.log_data['ts']))
		print('Score: {}'.format(self.log_data['score']))
		print('Timesteps per episode: {}'.format(self.log_data['t']))
		if 'avg_score' in self.log_types:
			print('Avg Score: {}'.format(self.log_data['avg_score']))
		if 'highest_score' in self.log_types:
			print('Highest Score: {}'.format(self.log_data['high_score']))
		if 'lowest_score' in self.log_types:
			print('Lowest Score: {}'.format(self.log_data['low_score']))
		print('{:-^50}'.format(''))
		# input()

	def log_state(self,t,score):
		self.update_state(t,score)		
		self.show_state()

	def update_state(self,t,score):
		self.log_data['epoch'] += 1
		self.log_data['ts'] += t
		self.log_data['score'] = score
		self.log_data['t'] = t
		self.log_data['avg_score'] = (self.log_data['avg_score'] * (self.log_data['epoch']-1) + self.log_data['score']) / self.log_data['epoch']
		self.log_data['high_score'] = max(self.log_data['score'],self.log_data['high_score'])
		self.log_data['low_score'] = min(self.log_data['score'],self.log_data['low_score'])

		if self.log_data['epoch'] % self.save_interval == 0:
			self.save_state()