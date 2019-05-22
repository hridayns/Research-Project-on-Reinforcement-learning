# ------------------ FOR REPRODUCABILITY OF RESULTS ---------------------- #
global_seed_val = 1

import os
os.environ['PYTHONHASHSEED'] = str(global_seed_val)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import random
random.seed(global_seed_val)

import numpy as np
np.random.seed(global_seed_val)

import tensorflow as tf
tf.set_random_seed(global_seed_val)

from tensorflow.python.keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# ------------------------------------------------------------------------ #

import gym
from gym.spaces.prng import seed
from collections import deque
import matplotlib.pyplot as plt
import argparse
import h5py
from timeit import default_timer as timer
from atari_wrappers import make_atari,wrap_deepmind

# from scipy.interpolate import spline

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten
from tensorflow.python.keras.optimizers import Adam

parser = argparse.ArgumentParser(description='Parse parameters.')

#TRAINING AND ENV PARAMS

# parser.add_argument('-eps','--episode_lim',help='episodes limit. Default is 1000.',type=int,default=1000)

parser.add_argument('-env','--environment',help='Environment',type=str,default='PongNoFrameskip-v4')
parser.add_argument('-ts','--ts_lim',help='Timestep limit. Default is 10M.',type=int,default=10000000)
parser.add_argument('-eps','--ep_lim',help='Episode limit. Default is 500.',type=int,default=500)
parser.add_argument('-plt_freq','--plot_frequency',help='Plot frequency. Default is 100.',type=int,default=100)
parser.add_argument('-window','--score_window_size',help='Score window size. Default is 100.',type=int,default=100)
parser.add_argument('-smooth','--smoothing_factor',help='Smoothing factor. Default is 2',type=int,default=2)
parser.add_argument('-save_freq','--save_frequency',help='Saving frequency (in eps). Default is 100.',type=int,default=100)
parser.add_argument('-mark_freq','--marking_freq',help='Mark every X',type=int,default=50)
parser.add_argument('-log_freq','--log_frequency',help='Log frequency. Default is 50.',type=int,default=50)
parser.add_argument('-plt_fn','--plot_folder_name',help='Plot folder name',type=str,default='')
# parser.add_argument('-max_ts','--max_timesteps',help='Maximum timesteps before done',type=int,default=10000)
parser.add_argument('-target','--target_score',help='Target score. Default is 19.0.',type=float,default=19.0)
parser.add_argument('-seed','--seed_nums', nargs='+', help='List of seed numbers to seed stochastic elements.', default=[1,12,23,34,42,51,69,72,87,98])
parser.add_argument('-drive','--drive_save',help='Dev argument for saving to drive; use when running on Google Collab.',action='store_true')


# RESEARCH PARAMS

parser.add_argument('-offset','--wave_offset',help='Wave offset. Default is 0.5.',type=float,default=0.5)
parser.add_argument('-anneal','--anneal_factor',help='Anneal factor. Default is 15%.',type=float,default=0.15)
parser.add_argument('-damp_freq','--damp_freq_factor',help='Damping frequency factor. Default is 1%.',type=float,default=0.01)

# HYPERPARAMETERS

parser.add_argument('-epsi_min','--epsilon_min',help='Minimum value of epsilon. Default is 0.001',type=float,default=0.001)

parser.add_argument('-replay','--buffer_size',help='Replay buffer size. Default is 10000.',type=int,default=10000)
parser.add_argument('-batch','--batch_size',help='Batch size. Default is 32.',type=int,default=32)
parser.add_argument('-alpha','--learning_rate',help='Learning rate. Default is 0.0001.',type=float,default=0.0001)
parser.add_argument('-gamma','--discount_factor',help='Discount Factor for rewards. Default is 0.99.',type=float,default=0.99)
parser.add_argument('-train_freq','--training_frequency',help='Model training frequency (in timesteps). Default is 4.',type=int,default=4)
parser.add_argument('-warm','--learn_start',help='Global timestep after which training can start. Default is 10000.',type=int,default=10000)
parser.add_argument('-target_update_freq','--target_update_frequency',help='Target update frequency. Default is 10.',type=int,default=1000)

#MC defaults
# batch_size = 64
# learning_rate = 0.0001
# gamma = 0.99
# target_train = 1

args = parser.parse_args()

seed_list = [int(x) for x in args.seed_nums]

ENV_NAME = args.environment
TIMESTEPS = args.ts_lim + 1
EPISODES = args.ep_lim + 1

if args.drive_save:
	root_path = '/content/drive/My Drive'
else:
	root_path = os.getcwd()

if args.plot_folder_name is not '':
	data_path = os.path.join(root_path,args.plot_folder_name)
else:
	env_part = ENV_NAME
	
	anneal_part = str(int(args.anneal_factor * 100))
	damp_part = str(args.damp_freq_factor * 100).replace('.','pt')
	param_part = anneal_part + '-' + damp_part

	buffer_part = str(int(args.buffer_size / 1000)) + 'K'

	smooth_part = 'smooth-' + str(args.smoothing_factor)

	data_path = os.path.join(root_path,'-'.join([env_part,param_part,buffer_part,smooth_part
		]))


class NN:
	def __init__(self,input_shape,num_actions,learning_rate=0.0001,blueprint=None):
		self.model = Sequential()
		
		for i in range(blueprint['conv_layers']):
			if i == 0:
				self.model.add(Conv2D(
					input_shape=input_shape,
					filters=blueprint['filters'][i],
					kernel_size=blueprint['kernel_sizes'][i],
					strides=blueprint['strides'][i],
					padding=blueprint['paddings'][i],
					activation=blueprint['activations'][i]
				))
			else:
				self.model.add(Conv2D(
					filters=blueprint['filters'][i],
					kernel_size=blueprint['kernel_sizes'][i],
					strides=blueprint['strides'][i],
					padding=blueprint['paddings'][i],
					activation=blueprint['activations'][i]
				))

		self.model.add(Flatten())

		self.model.add(Dense(
			units=blueprint['dense_units'],
			activation=blueprint['dense_activation']
		))

		self.model.add(Dense(
			units=num_actions
		))

		self.model.compile(
			loss=self.huber_loss,
			optimizer=Adam(
				lr=learning_rate
			),
			metrics=['acc']
		)

	def huber_loss(self,y_true, y_pred):
		return tf.losses.huber_loss(y_true,y_pred)

class ReplayBuffer:
	def __init__(self,run_path,buffer_size=10000,obs_shape=(84,84,4)):
		self.buffer_size = buffer_size
		self.replay_buffer_save_path = os.path.join(run_path,'replay-buffer.hdf5')

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

class ExplorationFactor:
	def __init__(self,tot_steps=int(1e7),initial_e=1.0,final_e=0.01):
		self.tot_steps = tot_steps
		self.initial_e = initial_e
		self.final_e = final_e
	def value(self,e):
		fract = min(float(e)/self.tot_steps , self.initial_e)
		return self.initial_e + fract * (self.final_e - self.initial_e)

	def mid_damp_value(self,x):
		return 0.5 + 0.5 * np.exp(-x/10) * np.cos(5*x)
		# y\ =\ 0.5\ +\ 0.5e^{-\frac{x}{10}}\cos\left(5x\right)
		# y\ =\ 0.5e^{-\frac{x}{10}}\left(1\ +\ \cos\left(5x\right)\right)
	def low_damp_value(self,x,wave_offset=0.5,anneal_factor=0.15,damp_freq_factor=0.01):
		return max(0.5 * np.exp(-x / anneal_factor) * (1 + np.cos(x / damp_freq_factor )), self.final_e)

class DDQN:
	def __init__(self,env,run_path):
		self.env = env

		self.blueprint = {
			'conv_layers':3,
			'filters': [32,64,64],
			'kernel_sizes': [(8,8),(4,4),(3,3)],
			'strides': [(4,4),(2,2),(1,1)],
			'paddings': ['valid','valid','valid'],
			'activations': ['relu','relu','relu'],
			'dense_units': 512,
			'dense_activation': 'relu'
		}

		self.train_targets = args.target_update_frequency # train target network every 'x' TIMESTEPS
		self.batch_size = args.batch_size # how much minimum memory before moving the weights
		self.gamma = args.discount_factor # discount factor for reward
		self.train_freq = args.training_frequency
		self.learn_start = args.learn_start
		self.epsilon_min = args.epsilon_min # epsilon must not decay below this
		self.epsilon = 1.0 # exploration vs exploitation factor
		self.exploration = ExplorationFactor(
			tot_steps = TIMESTEPS,
			initial_e = self.epsilon,
			final_e = self.epsilon_min
		)
		self.epsilon_decay = 0.995 #(self.epsilon - self.epsilon_min)/50000 # epsilon decays by this every episode or batch
		self.learning_rate = args.learning_rate

		self.input_shape = env.observation_space.shape
		self.num_actions = self.env.action_space.n
		self.replay_buffer = ReplayBuffer(run_path=run_path,buffer_size=args.buffer_size,obs_shape=self.input_shape)

		self.local_model = NN(input_shape = self.input_shape,num_actions = self.num_actions, learning_rate = self.learning_rate,blueprint=self.blueprint).model
		self.target_model = NN(input_shape = self.input_shape,num_actions = self.num_actions, learning_rate = self.learning_rate,blueprint=self.blueprint).model

		self.model_save_path = os.path.join(run_path,'local.h5')
		self.exploration_factor_save_path = os.path.join(run_path,'epsilon.npz')

		self.load()

	def save_mdl(self):
		self.local_model.save_weights(self.model_save_path)
		print('Local Model Saved due to mean score increase...')

	def load_mdl(self):
		if os.path.isfile(self.model_save_path):
			self.local_model.load_weights(self.model_save_path)
			print('Loaded Local Model...')
		else:
			print('No existing local model...')

	def save_exploration(self):
		np.savez(self.exploration_factor_save_path,exploration=self.epsilon)

	def load_exploration(self):
		if os.path.isfile(self.exploration_factor_save_path):
			with np.load(self.exploration_factor_save_path) as f:
				self.epsilon = np.asscalar(f['exploration'])
			print('Exploration Factor Loaded...')
		else:
			print('No existing Exploration Factor found...')

	def save(self):
		# self.save_mdl()
		self.save_exploration()
		self.replay_buffer.save()

	def load(self):
		self.load_mdl()
		self.load_exploration()
		self.replay_buffer.load()

	def choose_action(self,obs):
		if np.random.rand() < self.epsilon:
			return self.env.action_space.sample()
		q_vals = self.local_model.predict(np.expand_dims(obs,axis=0).astype(float)/255,batch_size=1)
		return np.argmax(q_vals[0])

	def update_epsilon(self,e,rt='eq 1'):
		if rt == 'eq 1':
			self.epsilon *= self.epsilon_decay
			self.epsilon = np.max([self.epsilon,self.epsilon_min])
		elif rt == 'eq 2':
			self.epsilon *= self.epsilon_decay
			self.epsilon = np.max([self.epsilon,self.epsilon_min])
		else:
			self.epsilon = self.exploration.low_damp_value(e,wave_offset=args.wave_offset,anneal_factor=args.anneal_factor * args.ep_lim,damp_freq_factor=args.damp_freq_factor * args.ep_lim)

	def remember(self,curr_obs,action,reward,next_obs,done):
		self.replay_buffer.add(curr_obs,action,reward,next_obs,done)

	def replay(self):
		if self.replay_buffer.meta_data['fill_size'] < self.batch_size:
			return

		curr_obs,action,reward,next_obs,done = self.replay_buffer.get_minibatch(self.batch_size)
		target = self.local_model.predict(curr_obs.astype(float)/255,batch_size=self.batch_size)

		done_mask = done.ravel()
		undone_mask = np.invert(done).ravel()

		target[done_mask,action[done_mask].ravel()] = reward[done_mask].ravel()

		Q_target = self.target_model.predict(next_obs.astype(float)/255,batch_size=self.batch_size)
		Q_future = np.max(Q_target[undone_mask],axis=1)

		target[undone_mask,action[undone_mask].ravel()] = reward[undone_mask].ravel() + self.gamma * Q_future

		self.local_model.fit(curr_obs.astype(float)/255, target, batch_size=self.batch_size, verbose=0).history

	def train_target_network(self):
		self.target_model.set_weights(self.local_model.get_weights())
		
def save_plot_data(plt_data,run_path):
	save_path = os.path.join(run_path,'plot_data.hdf5')
	print('Saving to {}...'.format(save_path))
	with h5py.File(save_path,'w') as f:
		for k in plt_data:
			dset = f.create_dataset(k,data=np.array(plt_data[k]))
		print('Plot Data Saved...')

def load_plot_data(run_path):
	save_path = os.path.join(run_path,'plot_data.hdf5')
	plot_data = {}
	load_flag = False
	if os.path.isfile(save_path):
		with h5py.File(save_path,'r') as f:
			for k in f:
				if k == 'score_window':
					plot_data[k] = deque(f[k],maxlen=args.score_window_size)
				else:
					plot_data[k] = deque(f[k])
				load_flag = True
	if load_flag:
		print('Plot Data loaded from {}...'.format(save_path))
		return plot_data

	print('No plot data found at {}...'.format(save_path))
	return None

def save_state(st_data,run_path):
	save_path = os.path.join(run_path,'state.npz')
	print('Saving to {}...'.format(save_path))
	np.savez(save_path,**st_data)
	print('State Saved...')

def load_state(run_path):
	save_path = os.path.join(run_path,'state.npz')
	if os.path.isfile(save_path):
		st_data = {}
		with np.load(save_path) as state:
			for k in state:
				st_data[k] = np.asscalar(state[k])	
		print('State loaded from {}...'.format(save_path))
		return st_data
	print('State not found at {} ...'.format(save_path))
	return None

def draw_plot(fig,ax,x,y,xlabel,ylabel,plot_name,plot_path,rt='eq 1',legend=False):
	x_smooth = np.array(x)[::args.smoothing_factor]
	y_smooth = np.array(y)[::args.smoothing_factor]

	if rt == 'eq 1':
		if not legend:
			ax.plot(x_smooth,y_smooth,'bo-',markevery=args.marking_freq)#,label='base')
		else:
			ax.plot(x_smooth,y_smooth,'bo-',markevery=args.marking_freq,label=rt)

	elif rt == 'eq 2':
		if not legend:
			ax.plot(x_smooth,y_smooth,'go-',markevery=args.marking_freq)#,label='novel')
		else:
			ax.plot(x_smooth,y_smooth,'go-',markevery=args.marking_freq,label=rt)
	else:
		if not legend:
			ax.plot(x_smooth,y_smooth,'ro--',markevery=args.marking_freq)#,label='novel')
		else:
			ax.plot(x_smooth,y_smooth,'ro--',markevery=args.marking_freq,label=rt)

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.grid(True)
	if legend:
		ax.legend()
	fig.savefig(os.path.join(plot_path,plot_name + '.png'),bbox_inches='tight')

def run(seed_num,seed_path,run_path,run_type,train=True):
	env = make_atari(ENV_NAME)
	env = wrap_deepmind(env, frame_stack=True, scale=False)
	env.seed(seed_num)
	seed(seed_num)

	dqn_agent = DDQN(env,run_path=run_path)
	dqn_agent.train_target_network()
	obs = env.reset().__array__(dtype=np.uint8)


	if train:
		
		state = load_state(run_path)

		if state is not None:
			ep = state['ep']
			t_start = state['t']
			ep_steps = state['ep_steps']
			# ep_score = state['score']
			std_dev_score = state['std_dev_score']
			dqn_agent.epsilon = state['epsilon']
			avg_score = state['avg_score']
			best_avg_score = state['best_avg_score']
			replay_fill_size = state['replay_fill_size']
		else:
			ep = 0
			t_start = 0
			ep_steps = 0
			replay_fill_size = 0
			# ep_score = 0
			std_dev_score = 0
			avg_score = -21
			best_avg_score = -21

		plt_data = load_plot_data(run_path)
		
		if plt_data is not None:
			avg_score_vals = plt_data['avg_score_vals']
			epsilon_vals = plt_data['epsilon_vals']
			best_avg_score_vals = plt_data['best_avg_score_vals']
			std_dev_score_vals = plt_data['std_dev_score_vals']
			replay_fill_size_vals = plt_data['replay_fill_size_vals']
			score_window = plt_data['score_window']
		else:
			avg_score_vals = deque()
			epsilon_vals = deque()
			# epsilon_vals.append(dqn_agent.epsilon)

			best_avg_score_vals = deque()
			std_dev_score_vals = deque()
			replay_fill_size_vals = deque()
			score_window = deque(maxlen=args.score_window_size)

			avg_score_vals.append(avg_score)
			epsilon_vals.append(dqn_agent.epsilon)
			best_avg_score_vals.append(best_avg_score)
			std_dev_score_vals.append(std_dev_score)
			replay_fill_size_vals.append(replay_fill_size)

		ep_score = 0

		max_score = -21
		min_score = 21


		print('avg_score_vals: {}'.format(avg_score_vals))
		print('avg_score_vals_size: {}'.format(len(avg_score_vals)))
		print('std_dev_score_vals: {}'.format(std_dev_score_vals))
		print('std_dev_score_vals size: {}'.format(len(std_dev_score_vals)))
		print('best_avg_score_vals: {}'.format(best_avg_score_vals))
		print('best_avg_score_vals size: {}'.format(len(best_avg_score_vals)))
		print('replay_fill_size_vals: {}'.format(replay_fill_size_vals))
		print('replay_fill_size_vals size: {}'.format(len(replay_fill_size_vals)))
		print('epsilon_vals: {}'.format(epsilon_vals))
		print('epsilon_vals size: {}'.format(len(epsilon_vals)))
		print('score_window: {}'.format(score_window))
		print('score_window size: {}'.format(len(score_window)))

		for t in range(t_start,TIMESTEPS):
			action = dqn_agent.choose_action(obs)
			dqn_agent.update_epsilon(ep,rt=run_type)
			new_obs, rew, done, _ = env.step(action)
			new_obs = new_obs.__array__(dtype=np.uint8)
			dqn_agent.remember(obs,action,rew,new_obs,done)
			obs = new_obs

			ep_score += rew
			ep_steps += 1

			if done:
				obs = env.reset().__array__(dtype=np.uint8)
				score_window.append(ep_score)

				ep += 1

				print('Episode {} | Timestep {} -> Score: {}'.format(ep,t,ep_score))

				avg_score = round(np.mean(score_window),1)
				if avg_score > best_avg_score and t > dqn_agent.learn_start and ep % args.save_frequency == 0:
					best_avg_score = avg_score
					dqn_agent.save_mdl()
				std_dev_score = round(np.std(score_window),1)
				avg_score_vals.append(avg_score)
				epsilon_vals.append(dqn_agent.epsilon)
				best_avg_score_vals.append(best_avg_score)
				std_dev_score_vals.append(std_dev_score)
				replay_fill_size_vals.append(dqn_agent.replay_buffer.meta_data['fill_size'])

				print('Size of avg_score_vals buffer: {}'.format(len(avg_score_vals)))

				if ep % args.log_frequency == 0:
					print('Avg score: {}'.format(avg_score))
					print('Time spent exploring: {} %'.format(round(100 * dqn_agent.exploration.low_damp_value(ep,wave_offset=args.wave_offset,anneal_factor=args.anneal_factor * args.ep_lim,damp_freq_factor=args.damp_freq_factor * args.ep_lim),2)))
					print('Std dev of score: {}'.format(std_dev_score))
					# print('Max score: {}'.format(max_score))
					# print('Min score: {}'.format(min_score))
					print('ReplayBuffer fill_size: {}'.format(dqn_agent.replay_buffer.meta_data['fill_size']))
					print('Score window contents: {}'.format(np.array(score_window)))

				if ep > 0 and ep % args.plot_frequency == 0:

					x_vals = np.arange(ep+1)
					draw_plot(avg_score_plt,ax1,x_vals,avg_score_vals,'Episodes','Avg Score',plot_name='Episodes vs Avg Score',plot_path=seed_path,rt=run_type)
					draw_plot(epsilon_vals_plt,ax2,x_vals,epsilon_vals,'Episodes','Epsilon',plot_name='Episodes vs Epsilon',plot_path=seed_path,rt=run_type)
					draw_plot(std_dev_plt,ax3,x_vals,std_dev_score_vals,'Episodes','Std Dev of Scores',plot_name='Episodes vs Std dev of score',plot_path=seed_path,rt=run_type)
					draw_plot(replay_fill_plt,ax4,x_vals,replay_fill_size_vals,'Episodes','Replay buffer Fill Size',plot_name='Episodes vs Replay Buffer Fill size',plot_path=seed_path,rt=run_type)
					draw_plot(best_avg_score_plt,ax5,x_vals,best_avg_score_vals,'Episodes','Best Avg Score',plot_name='Episodes vs Best Avg Score',plot_path=seed_path,rt=run_type)

				if ep % args.save_frequency == 0:
					plot_data = {
						'avg_score_vals': avg_score_vals,
						'epsilon_vals': epsilon_vals,
						'best_avg_score_vals': best_avg_score_vals,
						'std_dev_score_vals': std_dev_score_vals,
						'replay_fill_size_vals': replay_fill_size_vals,
						'score_window': score_window,
					}

					save_plot_data(plt_data=plot_data,run_path=run_path)
					state_data = {
						'ep':ep,
						't':t,
						'ep_steps': ep_steps,
						'score': ep_score,
						'avg_score' : avg_score,
						'std_dev_score': std_dev_score,
						'replay_fill_size': replay_fill_size,
						'best_avg_score': best_avg_score,
						'epsilon': dqn_agent.epsilon
					}
					save_state(st_data=state_data,run_path=run_path)
					dqn_agent.save()

				ep_score = 0
				ep_steps = 0


			if ep + 1 == EPISODES:# or best_avg_score >= args.target_score:
				print('---------------------------just before finish-----------------------------------------')
				x_vals = np.arange(ep+1)
				draw_plot(avg_score_plt,ax1,x_vals,avg_score_vals,'Episodes','Avg Score',plot_name='Episodes vs Avg Score',plot_path=seed_path,rt=run_type,legend=True)
				draw_plot(epsilon_vals_plt,ax2,x_vals,epsilon_vals,'Episodes','Epsilon',plot_name='Episodes vs Epsilon',plot_path=seed_path,rt=run_type,legend=True)
				draw_plot(std_dev_plt,ax3,x_vals,std_dev_score_vals,'Episodes','Std Dev of Scores',plot_name='Episodes vs Std dev of score',plot_path=seed_path,rt=run_type,legend=True)
				draw_plot(replay_fill_plt,ax4,x_vals,replay_fill_size_vals,'Episodes','Replay buffer Fill Size',plot_name='Episodes vs Replay Buffer Fill size',plot_path=seed_path,rt=run_type,legend=True)
				draw_plot(best_avg_score_plt,ax5,x_vals,best_avg_score_vals,'Episodes','Best Avg Score',plot_name='Episodes vs Best Avg Score',plot_path=seed_path,rt=run_type,legend=True)

				break

			if t > dqn_agent.learn_start:
				if t % dqn_agent.train_freq == 0:
					dqn_agent.replay()
				if t % dqn_agent.train_targets == 0:
					dqn_agent.train_target_network()

				# ax1.lines[-1].set_label(run_type)
				# ax2.lines[-1].set_label(run_type)
				# ax3.lines[-1].set_label(run_type)
				# ax4.lines[-1].set_label(run_type)
				# ax5.lines[-1].set_label(run_type)
				# ax1.legend()
				# ax2.legend()
				# ax3.legend()
				# ax4.legend()
				# ax5.legend()

if __name__ == '__main__':
	for s in seed_list:
		# plt.figure()
		avg_score_plt,ax1 = plt.subplots()
		epsilon_vals_plt,ax2 = plt.subplots()
		std_dev_plt,ax3 = plt.subplots()
		replay_fill_plt,ax4 = plt.subplots()
		best_avg_score_plt,ax5 = plt.subplots()
		print('Seed: {}'.format(s))

		seed_path = os.path.join(data_path,'Seed ' + str(s))

		if not os.path.exists(seed_path):
			os.makedirs(seed_path)


		rts = ['eq 1','proposed']

		run_state_path = os.path.join(seed_path,'run_state.npy')
		rt_st = 0

		if os.path.isfile(run_state_path):
			rt_st = int(np.load(run_state_path))
			print('Loaded Run state: {}...'.format(rts[rt_st]))
		else:
			print('No existing Run state found...')

		if rt_st == 1:
			print('-----ATTEMPTING TO DRAW eq 1 plots--------------')

			plot_resume_rt = 'eq 1'
			plot_resume_path = os.path.join(seed_path,plot_resume_rt)

			if os.path.isfile(os.path.join(plot_resume_path,'plot_data.hdf5')):
				plt_state = load_state(plot_resume_path)
				if plt_state is not None:
					x_vals = np.arange(int(plt_state['ep'])+1)
				else:
					print('Plot state resume failed...')

				plt_data = load_plot_data(plot_resume_path)
				if plt_data is not None:
					draw_plot(avg_score_plt,ax1,x_vals,plt_data['avg_score_vals'],'Episodes','Avg Score',plot_name='Episodes vs Avg Score',plot_path=seed_path,rt=plot_resume_rt,legend=True)
					draw_plot(epsilon_vals_plt,ax2,x_vals,plt_data['epsilon_vals'],'Episodes','Epsilon',plot_name='Episodes vs Epsilon',plot_path=seed_path,rt=plot_resume_rt,legend=True)
					draw_plot(std_dev_plt,ax3,x_vals,plt_data['std_dev_score_vals'],'Episodes','Std Dev of Scores',plot_name='Episodes vs Std dev of score',plot_path=seed_path,rt=plot_resume_rt,legend=True)
					draw_plot(replay_fill_plt,ax4,x_vals,plt_data['replay_fill_size_vals'],'Episodes','Replay buffer Fill Size',plot_name='Episodes vs Replay Buffer Fill size',plot_path=seed_path,rt=plot_resume_rt,legend=True)
					draw_plot(best_avg_score_plt,ax5,x_vals,plt_data['best_avg_score_vals'],'Episodes','Best Avg Score',plot_name='Episodes vs Best Avg Score',plot_path=seed_path,rt=plot_resume_rt,legend=True)
				else:
					print('Plot data resume failed...')

		for rt in range(rt_st,len(rts)):
			print('Run type: {}'.format(rts[rt]))
			run_path = os.path.join(seed_path,rts[rt])
			if not os.path.exists(run_path):
				os.makedirs(run_path)
			np.save(run_state_path,rt)
			print('Saving Run state...')
			run(seed_num=s,seed_path=seed_path,run_path=run_path,run_type=rts[rt])

		plt.close()

	tot_avg_plt_ep,ax_tot_ep = plt.subplots()
	# tot_avg_plt_t,ax_tot_t = plt.subplots()
	
	for rt in ['eq 1','proposed']:
		tot_avg = np.zeros(EPISODES)
		for s in seed_list:
			seed_path = os.path.join(data_path,'Seed ' + str(s))
			run_path = os.path.join(seed_path,rt)

			plot_data_path = os.path.join(run_path,'plot_data.hdf5')
			if os.path.isfile(plot_data_path):
				with h5py.File(plot_data_path,'r') as f:
					tot_avg += np.array(f['avg_score_vals'])

		tot_avg /= len(seed_list)
		# ax_tot.lines[-1].set_label(rt)
		# ax_tot.legend()

		# draw_plot(tot_avg_plt_t,ax_tot_t,np.arange(TIMESTEPS),tot_avg,'Timestep','Avg Score','Timesteps vs Avg Score',run_path=data_path,rt=rt,legend=True)
		draw_plot(tot_avg_plt_ep,ax_tot_ep,np.arange(EPISODES),tot_avg,'Episode','Avg Score','Episodes vs Avg Score',plot_path=data_path,rt=rt,legend=True)

	plt.close()


	# 1 12 23 34 42 51 69 72 87 98