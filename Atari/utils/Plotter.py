import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import h5py
from collections import deque

class Plotter:
	def __init__(self,env_name,plot_types=['avg_scores_ep','avg_scores_ts','scores_ep','scores_ts','high_scores_ep','high_scores_ts','low_scores_ep','low_scores_ts','timesteps_ep'],interval_types=['overall','window'],plot_interval=100,data_paths=None):
		self.env_name = env_name
		self.plot_types = plot_types
		self.plot_interval = plot_interval

		self.plots_save_folder = data_paths.save_folders['plot']
		self.plot_data_save_path = os.path.join(data_paths.save_folders['plot_data'],'plot_data.hdf5')

		self.epoch = 0
		self.interval_types = interval_types

		self.plot_data = {
			'scores': deque(),
			'avg_scores' : deque(),
			'high_scores' : deque(),
			'low_scores' : deque(),
			'ts': deque(),
			'tss': deque()
		}
		self.plot_data_dict = {
			'avg_scores_ep' : {
				'xlabel': 'Epoch',
				'ylabel': 'Avg Score',
				'metric': 'avg_scores',
				'vs': 'epoch'
			},
			'avg_scores_ts' : {
				'xlabel': 'Timesteps',
				'ylabel': 'Avg Score',
				'metric': 'avg_scores',
				'vs': 'timesteps'
			},
			'scores_ep' : {
				'xlabel': 'Epoch',
				'ylabel': 'Score',
				'metric': 'scores',
				'vs': 'epoch'
			},
			'scores_ts' : {
				'xlabel': 'Timesteps',
				'ylabel': 'Score',
				'metric': 'scores',
				'vs': 'timesteps'
			},
			'high_scores_ep' : {
				'xlabel': 'Epoch',
				'ylabel': 'Highest Score',
				'metric': 'high_scores',
				'vs': 'epoch'
			},
			'high_scores_ts' : {
				'xlabel': 'Timesteps',
				'ylabel': 'Highest Score',
				'metric': 'high_scores',
				'vs': 'timesteps'
			},
			'low_scores_ep' : {
				'xlabel': 'Epoch',
				'ylabel': 'Lowest Score',
				'metric': 'low_scores',
				'vs': 'epoch'
			},
			'low_scores_ts' : {
				'xlabel': 'Timesteps',
				'ylabel': 'Lowest Score',
				'metric': 'low_scores',
				'vs': 'timesteps'
			},
			'timesteps_ep' : {
				'xlabel': 'Epoch',
				'ylabel': 'Timesteps per episode',
				'metric': 'ts',
				'vs': 'epoch'
			}
		}

		self.load_plot_data()

	def save_plot_data(self):
		with h5py.File(self.plot_data_save_path,'w') as f:
			for k in self.plot_data:
				dset = f.create_dataset(k,data=np.asarray(self.plot_data[k]))
			print('Plot Data Saved...')
		
	def load_plot_data(self):
		load_flag = False
		if os.path.isfile(self.plot_data_save_path):
			with h5py.File(self.plot_data_save_path,'r') as f:
				for k in self.plot_data:
					self.plot_data[k] = deque(f[k])
					load_flag = True
		if load_flag:
			print('Plot Data Loaded...')
		else:
			print('No existing plot data found...')

	def update_plot_data(self,log_data):
		self.epoch = log_data['epoch']
		self.plot_data['ts'].append(log_data['t'])
		self.plot_data['tss'].append(log_data['ts'])
		self.plot_data['scores'].append(log_data['score'])
		self.plot_data['avg_scores'].append(log_data['avg_score'])
		self.plot_data['high_scores'].append(log_data['high_score'])
		self.plot_data['low_scores'].append(log_data['low_score'])


	def save_plot(self,x,y,xlabel,ylabel,plot_save_path):
		plt.figure()
		plt.plot(x,y,'bo-',markevery=self.plot_interval/5)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.savefig(plot_save_path,bbox_inches='tight')
		plt.close()

	def plot_graph(self,log_data):
		self.update_plot_data(log_data)
		# return
		if self.epoch % self.plot_interval == 0:
			self.save_plot_data()
			print('Drawing plots...')
			for pt in self.plot_types:
				if pt in self.plot_data_dict:
					xlabel = self.plot_data_dict[pt]['xlabel']
					ylabel = self.plot_data_dict[pt]['ylabel']
					y = np.asarray(self.plot_data[self.plot_data_dict[pt]['metric']])
					vs = self.plot_data_dict[pt]['vs']
					end = self.epoch+1
					plot_save_folder = None
					plot_name = ''
					x = None

					for interval in self.interval_types:
						plot_save_folder = os.path.join(self.plots_save_folder,*[interval,ylabel])
						
						if interval == 'overall':
							start = 1
							plot_name = xlabel

							if vs == 'epoch':
								x = np.arange(start,end)
							elif vs == 'timesteps':
								x = np.asarray(self.plot_data['tss'])
						else:
							start = self.epoch-(self.plot_interval-1)
							plot_name = xlabel + '_'
							plot_save_folder = os.path.join(plot_save_folder,vs)
							y = y[start-1:end]
							if vs == 'epoch':
								x = np.arange(start,end)
								plot_name += str(start) + '-' + str(end)
							elif vs == 'timesteps':
								x = np.asarray(self.plot_data['tss'])
								x = x[start-1:end]
								plot_name += str(x[0]) + '-' + str(x[-1])

						plot_name += '.png'
						if not os.path.exists(plot_save_folder):
							os.makedirs(plot_save_folder)
						plot_save_path = os.path.join(plot_save_folder,plot_name)
						x_size = len(x)
						y_size = len(y)

						if x_size == y_size:
							print('Plotting {} type plot: {} vs {}...'.format(interval,xlabel,ylabel))
							self.save_plot(x,y,xlabel,ylabel,plot_save_path)
						else:
							print('Mismatched sizes {} and  {} of x and y data...'.format(x_size,y_size))
							print('Skipping {} type plot: {} vs {}...'.format(interval,xlabel,ylabel))

			print('Finished drawing plots...')