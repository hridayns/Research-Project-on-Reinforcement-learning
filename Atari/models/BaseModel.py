

class BaseModel(object):
	def __init__(self,input_shape,num_actions,save_dirs):
		self.input_shape = input_shape
		self.num_actions = num_actions
		self.save_path = save_dirs['checkpoints']

	def step_update(self,tot_step):
		pass

	def act(self,obs):
		pass

	def remember(self,curr_obs,action,reward,next_obs,done):
		pass

	def update_exploration(self,t):
		pass

	def save(self,t,logger):
		pass