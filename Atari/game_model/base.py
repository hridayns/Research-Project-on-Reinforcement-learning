import os

class BaseGameModel:
	def __init__(self,game_name,input_dims,action_space,data_paths=None):
		self.input_dims = input_dims
		self.action_space = action_space
		self.model_path = data_paths.save_folders['model']

	def step_update(self,tot_step):
		pass

	def act(self,obs):
		pass

	def remember(self,curr_obs,action,reward,next_obs,done):
		pass
