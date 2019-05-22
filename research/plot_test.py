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
# from scipy.interpolate import spline

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import Adam

parser = argparse.ArgumentParser(description='Parse parameters.')

#TRAINING AND ENV PARAMS

parser.add_argument('-eps','--episode_lim',help='Episodes limit. Default is 1000.',type=int,default=1000)
parser.add_argument('-plt_freq','--plot_frequency',help='Plot frequency. Default is 100.',type=int,default=100)
parser.add_argument('-smooth','--smoothing_factor',help='Smoothing factor. Default is 2',type=int,default=2)
parser.add_argument('-mark_freq','--marking_freq',help='Mark every X',type=int,default=50)
parser.add_argument('-log_freq','--log_frequency',help='Log frequency. Default is 50.',type=int,default=50)
parser.add_argument('-plt_fn','--plot_folder_name',help='Plot folder name',type=str,default='')
parser.add_argument('-env','--environment',help='Environment',type=str,default='CartPole-v1')
parser.add_argument('-max_ts','--max_timesteps',help='Maximum timesteps before done',type=int,default=10000)
parser.add_argument('-seed','--seed_nums', nargs='+', help='List of seed numbers to seed stochastic elements.', default=[1,12,23,34,42,51,69,72,87,98])
parser.add_argument('-drive','--drive_save',help='Dev argument for saving to drive; use when running on Google Collab.',action='store_true')


# RESEARCH PARAMS

parser.add_argument('-offset','--wave_offset',help='Wave offset. Default is 0.5.',type=float,default=0.5)
parser.add_argument('-anneal','--anneal_factor',help='Anneal factor. Default is 15%.',type=float,default=0.15)
parser.add_argument('-damp_freq','--damp_freq_factor',help='Damping frequency factor. Default is 1%.',type=float,default=0.01)

# HYPERPARAMETERS

parser.add_argument('-epsi_min','--epsilon_min',help='Minimum value of epsilon. Default is 0.001',type=float,default=0.001)

parser.add_argument('-replay','--buffer_size',help='Replay buffer size. Default is 100000.',type=int,default=100000)
parser.add_argument('-batch','--batch_size',help='Batch size. Default is 20.',type=int,default=20)
parser.add_argument('-alpha','--learning_rate',help='Learning rate. Default is 0.001.',type=float,default=0.001)
parser.add_argument('-gamma','--discount_factor',help='Discount Factor for rewards. Default is 0.95.',type=float,default=0.95)
parser.add_argument('-target_update_freq','--target_update_frequency',help='Target update frequency. Default is 10.',type=int,default=10)

#MC defaults
# batch_size = 64
# learning_rate = 0.0001
# gamma = 0.99
# target_train = 1

args = parser.parse_args()

seed_list = [int(x) for x in args.seed_nums]

ENV_NAME = args.environment
EPISODES = args.episode_lim + 1

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
	def __init__(self,input_shape,num_actions,learning_rate = 0.01):
		self.input_shape = input_shape[0]
		self.num_actions = num_actions

		self.learning_rate = learning_rate

		self.model = Sequential()
		# self.model.add(Dense(8, input_dim=self.input_shape, activation="relu"))
		self.model.add(Dense(256, input_dim=self.input_shape, activation="relu"))
		self.model.add(Dense(256, activation="relu"))
		self.model.add(Dense(self.num_actions))
		self.model.compile(loss=self.huber_loss,optimizer=Adam(lr=self.learning_rate),metrics=['acc'])

	def huber_loss(self,y_true, y_pred):
		return tf.losses.huber_loss(y_true,y_pred)

class ReplayBuffer:
	def __init__(self,save_dirs=None,buffer_size=10000,obs_shape=(84,84,4)):
		self.buffer_size = buffer_size
		# self.replay_buffer_save_path = os.path.join(save_dirs['checkpoints'],'replay-buffer.hdf5')

		self.data = {
			'curr_obs': np.empty(shape=(self.buffer_size,obs_shape[0])),
			'action': np.empty(shape=(self.buffer_size,1),dtype=np.uint8),
			'reward': np.empty(shape=(self.buffer_size,1)),
			'next_obs': np.empty(shape=(self.buffer_size,obs_shape[0])),
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

class ExplorationFactor:
	def __init__(self,tot_eps=1000,initial_e=1.0,final_e=0.01):
		self.tot_eps = tot_eps
		self.initial_e = initial_e
		self.final_e = final_e
	def value(self,e):
		fract = min(float(e)/self.tot_eps , self.initial_e)
		return self.initial_e + fract * (self.final_e - self.initial_e)

	def mid_damp_value(self,x):
		return 0.5 + 0.5 * np.exp(-x/10) * np.cos(5*x)
		# y\ =\ 0.5\ +\ 0.5e^{-\frac{x}{10}}\cos\left(5x\right)
		# y\ =\ 0.5e^{-\frac{x}{10}}\left(1\ +\ \cos\left(5x\right)\right)
	def low_damp_value(self,x,wave_offset=0.5,anneal_factor=0.15,damp_freq_factor=0.01):
		return max(0.5 * np.exp(-x / anneal_factor) * (1 + np.cos(x / damp_freq_factor )), self.final_e)

class DDQN:
	def __init__(self,env):
		self.env = env

		self.train_targets = args.target_update_frequency # train target network every 'x' episodes
		self.batch_size = args.batch_size # how much minimum memory before moving the weights
		self.gamma = args.discount_factor # discount factor for reward
		self.epsilon_min = args.epsilon_min # epsilon must not decay below this
		self.epsilon = 1.0 # exploration vs exploitation factor
		self.exploration = ExplorationFactor(
			tot_eps = EPISODES,
			initial_e = self.epsilon,
			final_e = self.epsilon_min
		)
		self.epsilon_decay = 0.995 #(self.epsilon - self.epsilon_min)/50000 # epsilon decays by this every episode or batch
		self.learning_rate = args.learning_rate

		self.input_shape = env.observation_space.shape
		self.num_actions = self.env.action_space.n
		self.replay_buffer = ReplayBuffer(buffer_size=args.buffer_size,obs_shape=self.input_shape)

		self.local_model = NN(input_shape = self.input_shape,num_actions = self.num_actions, learning_rate = self.learning_rate).model
		self.target_model = NN(input_shape = self.input_shape,num_actions = self.num_actions, learning_rate = self.learning_rate).model

		# self.load_mdl()

	def save_mdl(self,trial_seed_path,rt='base'):
		self.local_model.save_weights(os.path.join(trial_seed_path,str(rt)+'-local.h5'))
		print('Local Model Saved due to mean score increase...')

	# def load_mdl(self,rt='base'):
	# 	if os.path.isfile(os.path.join(data_path,'local.h5')):
	# 		self.local_model.load_weights(os.path.join(data_path,'local.h5'))
	# 		print('Loaded Local Model...')
	# 	else:
	# 		print('No existing local model...')

	def perform_action(self,obs):
		qvals = self.model.forward_pass(obs)[0]
		return np.argmax(qvals)

	def choose_action(self,obs):
		if np.random.rand() < self.epsilon:
			return self.env.action_space.sample()
		qvals = self.local_model.predict(obs)[0]
		return np.argmax(qvals)

	def update_epsilon(self,e,rt='base'):
		if rt == 'eq 1':
			self.epsilon *= self.epsilon_decay
			self.epsilon = np.max([self.epsilon,self.epsilon_min])
		elif rt == 'eq 2':
			self.epsilon *= self.epsilon_decay
			self.epsilon = np.max([self.epsilon,self.epsilon_min])
		else:
			self.epsilon = self.exploration.low_damp_value(e,wave_offset=args.wave_offset,anneal_factor=args.anneal_factor * EPISODES,damp_freq_factor=args.damp_freq_factor * EPISODES)

	def remember(self,curr_obs,action,reward,next_obs,done):
		self.replay_buffer.add(curr_obs,action,reward,next_obs,done)

	def replay(self):
		if self.replay_buffer.meta_data['fill_size'] < self.batch_size:
			return

		curr_obs,action,reward,next_obs,done = self.replay_buffer.get_minibatch(self.batch_size)
		target = self.local_model.predict(curr_obs,batch_size=self.batch_size)

		done_mask = done.ravel()
		undone_mask = np.invert(done).ravel()

		target[done_mask,action[done_mask].ravel()] = reward[done_mask].ravel()

		Q_target = self.target_model.predict(next_obs,batch_size=self.batch_size)
		Q_future = np.max(Q_target[undone_mask],axis=1)

		target[undone_mask,action[undone_mask].ravel()] = reward[undone_mask].ravel() + self.gamma * Q_future

		hist = self.local_model.fit(curr_obs, target, batch_size=self.batch_size, verbose=0).history
		return hist

	def train_target_network(self):
		self.target_model.set_weights(self.local_model.get_weights())
		


def reshape_input(X):
	X = X.reshape(-1,X.shape[0])
	return X

def save_plot_data(d,trial_seed_path,rt):
	with h5py.File(os.path.join(trial_seed_path,rt+'_avg_data.h5'),'w') as f:
		f.create_dataset('avg_scores',data=np.array(d))

def save_plot(fig,ax,x,y,xlabel,ylabel,plot_name,trial_seed_path,rt='base',legend=False):
	x_smooth = np.array(x)[::args.smoothing_factor]
	y_smooth = np.array(y)[::args.smoothing_factor]
	# x_np = np.array(x)
	# y_np = np.array(y)
	# print(x_np)
	# x_smooth = np.linspace(x_np.min(), x_np.max(), 5)
	# print(x_smooth)
	# y_smooth = spline(x_np, y_np, x_smooth)

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
	fig.savefig(os.path.join(trial_seed_path,plot_name + '.png'),bbox_inches='tight')
	# plt.close()

def run(seed_num,train=True,run_type='base'):
	env = gym.make(ENV_NAME)#.env
	env._max_episode_steps = args.max_timesteps
	env.seed(seed_num)
	seed(seed_num)

	dqn_agent = DDQN(env)

	trial_seed_path = os.path.join(data_path,'Seed ' + str(seed_num))
	if not os.path.exists(trial_seed_path):
		os.makedirs(trial_seed_path)

	if train:
		max_score = -1000
		min_score = 1000
		avg_score = -1000
		best_avg_score = -1000

		epsilon_vals = deque(maxlen=EPISODES)
		avg_score_vals = deque(maxlen=EPISODES)
		best_avg_score_vals = deque(maxlen=EPISODES)
		# max_score_vals = deque(maxlen=EPISODES)
		# min_score_vals = deque(maxlen=EPISODES)
		std_dev_score_vals = deque(maxlen=EPISODES)
		replay_fill_size_vals = deque(maxlen=EPISODES)
		score_window = deque(maxlen=100)

		for ep in range(EPISODES):
			curr_obs = env.reset()
			curr_obs = reshape_input(curr_obs)
			total_r = 0
			if ep % 10 == 0:
				render = False#True
			epsilon_vals.append(dqn_agent.epsilon)
			while True:
				if render:
					env.render()
				action = dqn_agent.choose_action(curr_obs)
				next_obs,reward,done,info = env.step(action)
				next_obs = reshape_input(next_obs)
				total_r += reward
				if total_r % 1000 == 0:
					print('current reward for ep {}: reached {}'.format(ep,total_r))
				dqn_agent.remember(curr_obs,action,reward,next_obs,done)
				curr_obs = next_obs
				dqn_agent.replay()
				dqn_agent.update_epsilon(ep,run_type)
				
				if done:
					if render:
						render = False
						env.close()
					if ep % dqn_agent.train_targets == 0:
						dqn_agent.train_target_network()
					break
			score_window.append(total_r)
			avg_score = np.mean(score_window)
			if avg_score > best_avg_score:
				best_avg_score = avg_score
				dqn_agent.save_mdl(trial_seed_path,run_type)
			std_dev_score = np.std(score_window)
			# max_score = max(total_r,max_score)
			# min_score = min(total_r,min_score)
			# max_score_vals.append(max_score)
			# min_score_vals.append(min_score)
			avg_score_vals.append(avg_score)
			best_avg_score_vals.append(best_avg_score)
			std_dev_score_vals.append(std_dev_score)
			replay_fill_size_vals.append(dqn_agent.replay_buffer.meta_data['fill_size'])

			print('Episode ',ep,' -> Score: ',total_r)
			if ep % args.log_frequency == 0:
				print('Avg score: {}'.format(avg_score))
				print('Std dev of score: {}'.format(std_dev_score))
				print('Max score: {}'.format(max_score))
				print('Min score: {}'.format(min_score))
				print('ReplayBuffer fill_size: {}'.format(dqn_agent.replay_buffer.meta_data['fill_size']))

			if ep > 0 and ep % args.plot_frequency == 0:
				if ep + 1 == EPISODES:
					ax1.lines[-1].set_label(run_type)
					ax2.lines[-1].set_label(run_type)
					ax3.lines[-1].set_label(run_type)
					ax4.lines[-1].set_label(run_type)
					ax5.lines[-1].set_label(run_type)
					ax1.legend()
					ax2.legend()
					ax3.legend()
					ax4.legend()
					ax5.legend()

				x_vals = np.arange(ep+1)
				save_plot(avg_score_plt,ax1,x_vals,avg_score_vals,'Episodes','Avg Score',plot_name='Episodes vs Avg Score',trial_seed_path=trial_seed_path,rt=run_type)
				save_plot(epsilon_vals_plt,ax2,x_vals,epsilon_vals,'Episodes','Epsilon',plot_name='Episodes vs Epsilon',trial_seed_path=trial_seed_path,rt=run_type)
				save_plot(std_dev_plt,ax3,x_vals,std_dev_score_vals,'Episodes','Std Dev of Scores',plot_name='Episodes vs Std dev of score',trial_seed_path=trial_seed_path,rt=run_type)
				save_plot(replay_fill_plt,ax4,x_vals,replay_fill_size_vals,'Episodes','Replay buffer Fill Size',plot_name='Episodes vs Replay Buffer Fill size',trial_seed_path=trial_seed_path,rt=run_type)
				save_plot(best_avg_score_plt,ax5,x_vals,best_avg_score_vals,'Episodes','Best Avg Score',plot_name='Episodes vs Best Avg Score',trial_seed_path=trial_seed_path,rt=run_type)

				save_plot_data(d=avg_score_vals,trial_seed_path=trial_seed_path,rt=run_type)

if __name__ == '__main__':
	for s in seed_list:
		# plt.figure()
		avg_score_plt,ax1 = plt.subplots()
		epsilon_vals_plt,ax2 = plt.subplots()
		std_dev_plt,ax3 = plt.subplots()
		replay_fill_plt,ax4 = plt.subplots()
		best_avg_score_plt,ax5 = plt.subplots()
		print('Seed: {}'.format(s))
		for rt in ['eq 1','proposed']:
			print('Run type: {}'.format(rt))
			run(seed_num=s,run_type=rt)

		plt.close()

	tot_avg_plt,ax_tot = plt.subplots()
	for rt in ['eq 1','proposed']:
		tot_avg = np.zeros(EPISODES)
		for s in seed_list:
			trial_seed_path = os.path.join(data_path,'Seed ' + str(s))
			avg_data_path = os.path.join(trial_seed_path,rt+'_avg_data.h5')
			if os.path.isfile(avg_data_path):
				with h5py.File(avg_data_path,'r') as f:
					tot_avg += np.array(f['avg_scores'])

		tot_avg /= len(seed_list)
		# ax_tot.lines[-1].set_label(rt)
		# ax_tot.legend()

		save_plot(tot_avg_plt,ax_tot,np.arange(EPISODES),tot_avg,'Episode','Avg Score','Episodes vs Avg Score',trial_seed_path=data_path,rt=rt,legend=True)

	plt.close()