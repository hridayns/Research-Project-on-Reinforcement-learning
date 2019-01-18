import os

MODEL_PATH = os.path.join(os.getcwd(),'saved_models')

class BaseGameModel:
	def __init__(self,game_name,input_dims,action_space):
		self.input_dims = input_dims
		self.action_space = action_space
		self.model_path = os.path.join(MODEL_PATH,game_name)
		# print(self.model_path)

	def step_update(self,tot_step):
		pass

	def act(self,obs):
		pass

	def remember(self,curr_obs,action,reward,next_obs,done):
		pass
