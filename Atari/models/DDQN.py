import numpy as np
import os

from utils import ReplayBuffer
from NN import NeuralNet
from models import BaseModel

class LinearSchedule:
	def __init__(self,schedule_timesteps,final_p,initial_p=1.0):
		self.schedule_timesteps = schedule_timesteps
		self.final_p = final_p
		self.initial_p = initial_p
	def value(self,t):
		fraction = min(float(t)/self.schedule_timesteps , 1.0)
		return self.initial_p + fraction * (self.final_p - self.initial_p)

class DDQN(BaseModel):
	def __init__(self,env,save_dirs,learning_rate=0.0001):
		BaseModel.__init__(self,input_shape=env.observation_space.shape,num_actions=env.action_space.n,save_dirs=save_dirs)

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

		self.local_model_save_path = os.path.join(self.save_path,'local-wts.h5')
		self.local_model = NeuralNet(input_shape=self.input_shape,num_actions=self.num_actions,learning_rate=learning_rate,blueprint=self.blueprint).model

	def load_mdl(self):
		if os.path.isfile(self.local_model_save_path):
			self.local_model.load_weights(self.local_model_save_path)
			print('Loaded Local Model...')
		else:
			print('No existing Local Model found...')

	def load(self):
		self.load_mdl()

class DDQNLearner(DDQN):
	def __init__(self,env,save_dirs,save_freq=10000,gamma=0.99,batch_size=32,learning_rate=0.0001,buffer_size=10000,learn_start=10000,target_network_update_freq=1000,train_freq=4,epsilon_min=0.01,exploration_fraction=0.1,tot_steps=int(1e7)):
		DDQN.__init__(self,env=env,save_dirs=save_dirs,learning_rate=learning_rate)

		self.gamma = gamma
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.buffer_size = buffer_size
		self.learn_start = learn_start
		self.target_network_update_freq = target_network_update_freq
		self.train_freq = train_freq
		self.epsilon_min = epsilon_min
		self.exploration_fraction = exploration_fraction
		self.tot_steps = tot_steps
		self.epsilon = 1.0
		self.exploration = LinearSchedule(
			schedule_timesteps=int(self.exploration_fraction * self.tot_steps),
			initial_p=self.epsilon,
			final_p=self.epsilon_min
		)

		self.save_freq = save_freq

		self.replay_buffer = ReplayBuffer(save_dirs=save_dirs,buffer_size=self.buffer_size,obs_shape=self.input_shape)
		
		self.exploration_factor_save_path = os.path.join(self.save_path,'exploration-factor.npz')

		self.target_model_save_path = os.path.join(self.save_path,'target-wts.h5')
		self.target_model = NeuralNet(input_shape=self.input_shape,num_actions=self.num_actions,learning_rate=learning_rate,blueprint=self.blueprint).model

		self.show_hyperparams()

		self.update_target()

		self.load()

	def update_exploration(self,t):
		self.epsilon = self.exploration.value(t)

	def update_target(self):
		self.target_model.set_weights(self.local_model.get_weights())

	def remember(self,obs,action,rew,new_obs,done):
		self.replay_buffer.add(obs,action,rew,new_obs,done)

	def step_update(self,t):
		hist = None

		if t <= self.learn_start:
			return hist
		if t % self.train_freq == 0:
			hist = self.learn()
		if t % self.target_network_update_freq == 0:
			self.update_target()
		return hist

	def act(self,obs):
		if np.random.rand() < self.epsilon:
			return self.env.action_space.sample()
		q_vals = self.local_model.predict(np.expand_dims(obs,axis=0),batch_size=1)
		return np.argmax(q_vals[0])

	def learn(self):
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

		hist = self.local_model.fit(curr_obs.astype(float)/255, target, batch_size=self.batch_size, verbose=0).history
		return hist

	def load_mdl(self):
		super().load_mdl()
		if os.path.isfile(self.target_model_save_path):
			self.target_model.load_weights(self.target_model_save_path)
			print('Loaded Target Model...')
		else:
			print('No existing Target Model found...')

	def save_mdl(self):
		self.local_model.save_weights(self.local_model_save_path)
		print('Local Model Saved...')
		self.target_model.save_weights(self.target_model_save_path)
		print('Target Model Saved...')

	def save_exploration(self):
		np.savez(self.exploration_factor_save_path,exploration=self.epsilon)
		print('Exploration Factor Saved...')

	def load_exploration(self):
		if os.path.isfile(self.exploration_factor_save_path):
			with np.load(self.exploration_factor_save_path) as f:
				self.epsilon = np.asscalar(f['exploration'])
			print('Exploration Factor Loaded...')
		else:
			print('No existing Exploration Factor found...')


	def save(self,t,logger):
		ep = logger.data['episode']
		if (self.save_freq is not None and t > self.learn_start and ep > 100 and t % self.save_freq == 0):
			if logger.update_best_score():
				logger.save_state()
				self.save_mdl()
				self.save_exploration()
				self.replay_buffer.save()

	def load(self):
		self.load_mdl()
		self.load_exploration()
		self.replay_buffer.load()

	def show_hyperparams(self):
		print('Discount Factor (gamma): {}'.format(self.gamma))
		print('Batch Size: {}'.format(self.batch_size))
		print('Replay Buffer Size: {}'.format(self.buffer_size))
		print('Training Frequency: {}'.format(self.train_freq))
		print('Target network update Frequency: {}'.format(self.target_network_update_freq))
		print('Replay start size: {}'.format(self.learn_start))



class DDQNPlayer(DDQN):
	def __init__(self,env,save_dirs,learning_rate=0.0001,epsilon_test=0.02):
		DDQN.__init__(self,env=env,save_dirs=save_dirs,learning_rate=learning_rate)

		self.epsilon_test = epsilon_test

		self.load()

	def act(self,obs):
		if np.random.rand() < self.epsilon_test:
			return self.env.action_space.sample()
		q_vals = self.local_model.predict(np.expand_dims(obs,axis=0),batch_size=1)
		return np.argmax(q_vals[0])