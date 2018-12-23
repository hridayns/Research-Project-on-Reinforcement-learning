import gym
import numpy as np
import random
import sys

from collections import deque


np.random.seed(42)
env_name = 'MountainCar-v0'
saved_params_file =  env_name + '-DQN-saved-params.npz'
episodes = 5000
render = False

train = True
if(len(sys.argv) > 1):
	if sys.argv[1] == 'test':
		train = False


class NN:
	def __init__(self,input_size,output_size):
		self.hidden_size = 8
		self.alpha = 1e-2 # learning_rate
		self.lmbda = 0# regularization factor

		self.W = {
			'1' : np.random.randn(input_size,self.hidden_size),
			'2' : np.random.randn(self.hidden_size,output_size)
		}
		self.b = {
			'1' : np.ones((1,self.hidden_size)),
			'2' : np.ones((1,output_size))
		}

	def set_params(self,W,b):
		self.W = W
		self.b = b

	def get_wts(self):
		return self.W
	def get_biases(self):
		return self.b

	def show_wts(self):
		print('W1',self.W['1'])
		print('W2',self.W['2'])
		print('b1',self.b['1'])
		print('b2',self.b['2'])

	def forward_pass(self,X):
		hidden_val = np.dot(X,self.W['1']) + self.b['1']
		hidden_val = self.sigmoid(hidden_val)
		output_val = np.dot(hidden_val,self.W['2']) + self.b['2']

		return output_val,hidden_val

	def compute_gradient(self,X,y):
		y_hat,hidden = self.forward_pass(X)
		m = y.shape[0]
		err_at_l2 = (1/m) * (y_hat - y)# * self.sigmoid_der(y_hat)

		dL_Why = np.dot(hidden.T,err_at_l2)# + (self.lmbda/m) * self.W['2']
		dL_bhy = np.sum(err_at_l2,axis=0,keepdims=True)

		err_at_l1 = self.sigmoid_der(hidden) * np.dot(err_at_l2,self.W['2'].T)

		dL_Wxh = np.dot(X.T,err_at_l1)# + (self.lmbda/m) * self.W['1']
		dL_bxh = np.sum(err_at_l1,axis=0,keepdims=True)

		return {
			'W1' : dL_Wxh,
			'W2' : dL_Why,
			'b1' : dL_bxh,
			'b2' : dL_bhy,
		}

	def clip_grads(self,grad):
		low = -1
		high = 1
		grad['W1'] = np.clip(grad['W1'],low,high)
		grad['W2'] = np.clip(grad['W2'],low,high)
		return grad

	def update_wts(self,grad):
		self.W['1'] -= self.alpha * grad['W1']
		self.W['2'] -= self.alpha * grad['W2']
		self.b['1'] -= self.alpha * grad['b1']
		self.b['2'] -= self.alpha * grad['b2']

	def sigmoid(self,x):
		return 1.0/(1 + np.exp(-x))

	def sigmoid_der(self,x):
		return x * (1.0 - x)

	def relu(self,x):
		x[x<0]=0
		return x

	def relu_der(self,x):
		x[x<0] = 0
		x[x >= 0] = 1
		return x

	def tanh(self,x):
		e_2x = np.exp(-2*x)
		return (1.0 - e_2x)/(1.0 + e_2x)

	def tanh_der(self,x):
		return 1.0 - np.power(x,2)

class DQN:
	def __init__(self,env):
		self.env = env
		self.memory = deque(maxlen=100000)

		self.train_targets = 10 # train target network every 'x' episodes
		self.batch_size = 10 # how much minimum memory before moving the weights
		self.gamma = 0.95 # discount factor for reward
		self.epsilon = 1.0 # exploration vs exploitation factor
		self.epsilon_min = 0.01 # epsilon must not decay below this
		self.epsilon_decay = 0.995 #(self.epsilon - self.epsilon_min)/50000 # epsilon decays by this every episode or batch

		self.input_size = env.observation_space.shape[0]
		self.output_size = self.env.action_space.n

		self.model = self.init_NN()
		self.target_model = self.init_NN()

	def save_model(self,model):
		np.savez(saved_params_file,W1=model.W['1'],W2=model.W['2'],b1=model.b['1'],b2=model.b['2'])

	def load_model(self):
		model = self.model
		best_params = np.load(saved_params_file)
		W = {
			'1' : best_params['W1'],
			'2' : best_params['W2']
		}
		b = {
			'1' : best_params['b1'],
			'2' : best_params['b2'],
		}
		model.set_params(W,b)

	def init_NN(self):
		nn = NN(self.input_size,self.output_size)
		return nn

	def perform_action(self,obs):
		qvals = self.model.forward_pass(obs)[0]
		return np.argmax(qvals)

	def choose_action(self,obs):
		if np.random.rand() < self.epsilon:
			return self.env.action_space.sample()
		qvals = self.model.forward_pass(obs)[0]
		return np.argmax(qvals)

	def update_epsilon(self):
		self.epsilon *= self.epsilon_decay
		self.epsilon = np.max([self.epsilon,self.epsilon_min])

	def remember(self,curr_obs,action,reward,next_obs,done):
		self.memory.append([curr_obs,action,reward,next_obs,done])

	def replay(self):
		if len(self.memory) < self.batch_size:
			return
		mini_batch = random.sample(self.memory,self.batch_size)

		for i in range(self.batch_size):
			curr_obs,action,reward,next_obs,done = mini_batch[i]
			trueQ = reward

			if not done:
				trueQ = (reward + self.gamma * np.max(self.target_model.forward_pass(next_obs)[0]))
			
			Q_out = self.model.forward_pass(curr_obs)[0][0]
			Q_out[action] = trueQ

			grad = self.model.compute_gradient(curr_obs,Q_out)
			self.model.update_wts(grad)
		self.update_epsilon()


	def train_target_network(self):
		self.target_model.set_params(self.model.get_wts(),self.model.get_biases())

def reshape_input(X):
	X = X.reshape(-1,X.shape[0])
	return X

def play_game(agent):
	agent.load_model()
	env = agent.env
	curr_obs = env.reset()
	curr_obs = reshape_input(curr_obs)
	total_r = 0
	while True:
		env.render()
		action = agent.perform_action(curr_obs)
		next_obs,reward,done,info = env.step(action)
		next_obs = reshape_input(next_obs)
		total_r += reward

		curr_obs = next_obs
		if done:
			break
	print('Reward: ',total_r)

def run(train):
	env = gym.make(env_name)
	dqn_agent = DQN(env)

	if train:
		solution_count = 0
		count_1000 = 0
		count_2000 = 0
		count_3000 = 0
		count_4000 = 0
		count_5000 = 0
		count_6000 = 0
		for ep in range(episodes):
			curr_obs = env.reset()
			curr_obs = reshape_input(curr_obs)
			total_r = 0
			if ep % 100 == 0:
				render = True

			while True:
				if render:
					env.render()
				action = dqn_agent.choose_action(curr_obs)
				next_obs,reward,done,info = env.step(action)
				next_obs = reshape_input(next_obs)
				total_r += reward
				dqn_agent.remember(curr_obs,action,reward,next_obs,done)
				curr_obs = next_obs
				dqn_agent.replay()
			
				if done:
					if render:
						render = False
						env.close()
					if total_r > -200:
						solution_count += 1
						if ep <= 1000:
							count_1000 += 1
						elif ep <= 2000:
							count_2000 += 1
						elif ep <= 3000:
							count_3000 += 1
						elif ep <= 4000:
							count_4000 += 1
						elif ep <= 5000:
							count_5000 += 1
						else:
							count_6000 += 1
						print(solution_count,' solutions')
					if ep % dqn_agent.train_targets == 0:
						dqn_agent.train_target_network()
					break
			print('Episode ',ep,' -> Score: ',total_r)
		dqn_agent.save_model(dqn_agent.model)
		print('Total: ',solution_count)
		print('1 - 1000: ',count_1000)
		print('1000 - 2000: ',count_2000)
		print('2000 - 3000: ',count_3000)
		print('3000 - 4000: ',count_4000)
		print('4000 - 5000: ',count_5000)
		print('5000 - 6000: ',count_6000)
	else:
		play_game(dqn_agent)


if __name__ == '__main__':
	run(train)