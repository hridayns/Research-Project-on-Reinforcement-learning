import gym
import cv2 as cv
import numpy as np
from gym import spaces
from collections import deque

class FrameStack(gym.Wrapper):
	def __init__(self, env, k):
		gym.Wrapper.__init__(self, env)
		self.k = k
		self.frames = deque([], maxlen=k)
		shp = env.observation_space.shape
		self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

	def preprocess(self,obs):
		# obs = obs[::2,::2,::]
		obs = cv.resize(obs,(80,80))
		obs = cv.cvtColor(obs, cv.COLOR_RGB2GRAY)
		obs = obs[np.newaxis,...,np.newaxis]
		obs = np.divide(obs,255.0)

		return obs

	def reset(self):
		ob = self.env.reset()
		ob = self.preprocess(ob)
		for _ in range(self.k):
			self.frames.append(ob)
		return self._get_ob()

	def step(self, action):
		ob, reward, done, info = self.env.step(action)
		ob = self.preprocess(ob)
		self.frames.append(ob)
		return self._get_ob(), reward, done, info

	def _get_ob(self):
		assert len(self.frames) == self.k
		return LazyFrames(list(self.frames))

class LazyFrames(object):
	def __init__(self, frames):
		self._frames = frames
		self._out = None

	def _force(self):
		if self._out is None:
			self._out = np.concatenate(self._frames, axis=-1)
			self._frames = None
		return self._out

	def __array__(self, dtype=None):
		out = self._force()
		if dtype is not None:
			out = out.astype(dtype)
		return out

	def __len__(self):
		return len(self._force())

	def __getitem__(self, i):
		return self._force()[i]