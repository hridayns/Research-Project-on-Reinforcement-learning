import numpy as np
import gym
import cv2 as cv
import random
import tensorflow as tf
import arg_parser

from gym import spaces
from collections import deque
from pathlib import Path
from FrameStacker import FrameStack
# from sys import getsizeof

from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten
from tensorflow.python.keras.optimizers import RMSprop

ENV_NAME = 'Pong-v0'
LOCAL_NETWORK_WEIGHTS_SAVE = ENV_NAME + '-LOCAL-SAVED-WEIGHTS.h5'
TARGET_NETWORK_WEIGHTS_SAVE = ENV_NAME + '-TARGET-SAVED-WEIGHTS.h5'
CHECKPOINT_PARAMS_SAVE = ENV_NAME + '-CHECKPOINT-PARAMS.npz'

FRAME_BUFFER_SIZE = 4
EPISODES = 100
GAMMA = 0.99
ALPHA = 0.00025
REPLAY_MEM_SIZE = 50000
BATCH_SIZE = 32

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_STEPS = 850000
EXPLORATION_DECAY = (EXPLORATION_MAX-EXPLORATION_MIN)/EXPLORATION_STEPS

TRAINING_FREQUENCY = 4
TARGET_NETWORK_UPDATE_FREQUENCY = 40000
MODEL_SAVE_FREQUENCY = 100
REPLAY_START = 50000

EXPLORATION_TEST = 0.02

args = arg_parser.parse()

if args.episodes:
	EPISODES = args.episodes
if args.buffer_size:
	REPLAY_MEM_SIZE = args.buffer_size
if args.gamma:
	GAMMA = args.gamma
if args.lr:
	ALPHA = args.lr
if args.batch_size:
	BATCH_SIZE = args.batch_size
if args.test:
	#implement agent playing game using learned parameters
	print('Testing agent...')


class Learner:
	def __init__(self,env):
		self.env = env

		self.input_dims = self.env.reset().__array__()[0].shape
		self.output_dims = self.env.action_space.n

		self.k = FRAME_BUFFER_SIZE
		self.memory = deque(maxlen=REPLAY_MEM_SIZE)
		
		self.batch_size = BATCH_SIZE
		self.gamma = GAMMA
		self.alpha = ALPHA
		self.epsilon = EXPLORATION_MAX
		self.epsilon_min = EXPLORATION_MIN
		self.epsilon_decay = EXPLORATION_DECAY

		self.ep_start = 0
		self.training_freq = TRAINING_FREQUENCY
		self.replay_start = REPLAY_START
		self.model_save_freq = MODEL_SAVE_FREQUENCY
		self.target_update_freq = TARGET_NETWORK_UPDATE_FREQUENCY

		self.local_model = self.init_model()
		self.target_model = self.init_model()

		self.load_checkpoint()

		# self.target_train()

	def init_model(self):
		model = Sequential()
		model.add(Conv2D(input_shape=self.input_dims,filters=32,kernel_size=(8,8),strides=(4,4),padding='valid',activation='relu'))
		model.add(Conv2D(filters=64,kernel_size=(4,4),strides=(2,2),padding='valid',activation='relu'))
		model.add(Conv2D(filters=64,kernel_size=(2,2),strides=(1,1),padding='valid',activation='relu'))
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Dense(self.output_dims))
		model.compile(loss="mean_squared_error",optimizer=RMSprop(lr=self.alpha,rho=0.95,epsilon=0.01),metrics=["accuracy"])
		return model

	def load_checkpoint(self):
		print('in chkpoint load func')
		if Path(LOCAL_NETWORK_WEIGHTS_SAVE).exists():
			self.local_model.load_weights(LOCAL_NETWORK_WEIGHTS_SAVE)
			print('local model loaded')

		if Path(TARGET_NETWORK_WEIGHTS_SAVE).exists():
			self.target_model.load_weights(TARGET_NETWORK_WEIGHTS_SAVE)
			print('target model loaded')

		if Path(CHECKPOINT_PARAMS_SAVE).exists():
			params = np.load(CHECKPOINT_PARAMS_SAVE)
			self.ep_start = params['episode']
			self.epsilon = params['epsilon']
			params.close()

	def save_checkpoint(self):
		self.local_model.save_weights(LOCAL_NETWORK_WEIGHTS_SAVE)
		self.target_model.save_weights(TARGET_NETWORK_WEIGHTS_SAVE)

	def remember(self,curr_obs,action,reward,next_obs,done):
		self.memory.append([curr_obs,action,reward,next_obs,done])
		'''
		block = (curr_obs.nbytes + next_obs.nbytes + getsizeof(action) + getsizeof(reward) + getsizeof(done))/(1024*1024)
		n = len(self.memory)

		print('action size: {} bytes'.format(action.itemsize))
		print('reward size: {} bytes'.format(reward.itemsize))
		print('done size: {} bytes'.format(done.itemsize))

		print('One block = {} MB'.format(block))
		
		print('Memory spaces filled: {}/{} blocks'.format(n,REPLAY_MEM_SIZE))
		print('current size of memory: {}/{} MB'.format(block*n,block*REPLAY_MEM_SIZE))

		input()
		'''

	def step_update(self,tot_step):
		if(len(self.memory)) < self.replay_start:
			return
		if tot_step % self.training_freq == 0:
			self.replay()

		self.update_exploration()

		if tot_step	% self.model_save_freq == 0:
			self.save_checkpoint()

		if tot_step % self.target_update_freq == 0:
			self.target_train()

	def replay(self):
		if len(self.memory) < self.batch_size: 
			return

		samples = random.sample(self.memory, self.batch_size)

		update_input = np.zeros((self.batch_size,self.input_dims[0],self.input_dims[1],self.input_dims[2]))
		update_target = np.zeros((self.batch_size,self.output_dims))
		
		for i in range(self.batch_size):
			curr_obs, action, reward, next_obs, done = samples[i]
			target = self.local_model.predict(curr_obs)
			if done:
				target[0][action] = reward
			else:
				Q_future = np.max(self.target_model.predict(next_obs)[0])
				target[0][action] = reward + self.gamma * Q_future
			
			update_input[i] = curr_obs
			update_target[i] = target


		fit = self.local_model.fit(update_input, update_target, batch_size=self.batch_size, verbose=0)
		# loss = fit.history["loss"][0]
		# acc = fit.history["acc"][0]
		# print('Loss: {}, Acc: {}'.format(loss,acc))

		

	def act(self,obs):
		if np.random.random() < self.epsilon:
			return self.env.action_space.sample()
		q_vals = self.local_model.predict(obs)
		return np.argmax(q_vals[0])

	def update_exploration(self):
		self.epsilon = np.max([self.epsilon_min,self.epsilon - self.epsilon_decay])

	def target_train(self):
		self.target_model.set_weights(self.local_model.get_weights())

	def render_frame(self,obs):
		scale = 5
		window_width = int(obs.shape[1] * scale)
		window_height = int(obs.shape[0] * scale)
		cv.namedWindow('frame', cv.WINDOW_NORMAL)
		cv.resizeWindow('frame', window_width, window_height)

		cv.imshow('frame', obs[:,:,self.k-1])
		cv.waitKey(50)

env = gym.make(ENV_NAME)
env = FrameStack(env,FRAME_BUFFER_SIZE)

agent = Learner(env=env)

global_step = 0

print('Resuming from episode: ',agent.ep_start)

for ep in range(agent.ep_start,EPISODES):
	curr_obs = env.reset()
	curr_obs = curr_obs.__array__(dtype=np.float32)
	step = 0
	total_r = np.int8(0)
	render = False
	if args.render:
		if ep % args.render_freq == 0:
			render = True
	while True:
		if render:
			# agent.render_frame(curr_obs)
			env.render()
		global_step += 1
		step += 1
		action = agent.act(curr_obs)
		# memory saving step
		action = np.uint8(action)

		next_obs,reward,done,info = env.step(action)
		next_obs = next_obs.__array__(dtype=np.float32)
		
		# memory saving step
		reward = np.int8(reward)
		done = np.bool_(done)
		
		total_r += reward

		agent.remember(curr_obs,action,reward,next_obs,done)
		curr_obs = next_obs
		agent.step_update(global_step)
		if done:
			# agent.target_train()
			env.close()
			break

	np.savez(CHECKPOINT_PARAMS_SAVE,episode=ep,epsilon=agent.epsilon)
	print('Total Reward for episode {}: {}'.format(ep,total_r))