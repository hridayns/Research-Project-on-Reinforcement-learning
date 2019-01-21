# Internal Imports
from utils import arg_parser
from utils.AtariWrapper import GymAtari
from game_model.ddqn import DDQNLearner,DDQNPlayer

# External Imports
import os
import gym
import numpy as np
from collections import deque

args = arg_parser.parse()

class AtariRL:
	def __init__(self):
		self.args = args
		self.collab = self.args.collab_drive_save

		game_name = self.args.game_name
		mode = self.args.mode
		env_name = game_name + 'Deterministic-v4'
		env = GymAtari.wrap(gym.make(env_name),4)
		input_dims = env.reset().__array__().shape[1:]

		agent = self.get_agent(game_name,mode,input_dims,env.action_space)
		self.game_loop(agent,env,self.args.render,self.args.render_freq,self.args.episodes,self.args.total_step_lim,self.args.clip)

	def game_loop(self,agent,env,render,rf,episodes,total_step_lim,clip):
		
		agent.show_hyperparams()

		ep = 0
		total_step = 0
		window_len = 100
		score_window = deque(maxlen=window_len)

		if os.path.isfile(agent.state_save_path):
			with np.load(agent.state_save_path) as state:
				ep = state['ep']
				total_step = state['total_step']
				agent.epsilon = state['epsilon']
				print('Loaded state params: episode, total step, and epsilon...')

		if os.path.isfile(agent.score_buffer_save_path):
			with open(self.score_buffer_save_path, 'rb') as handle:
				score_window = pickle.load(handle)
				print('Score Buffer loaded...')

		print('Resuming from episode {}, global timestep {}...'.format(ep,total_step))

		while True:
			if ep >= episodes:
				print('Episode limit of {} episodes reached'.format(episodes))
				exit()
			ep += 1
			curr_obs = env.reset()
			curr_obs = curr_obs.__array__(dtype=np.float32)

			step = 0
			score = 0

			while True:
				if total_step >= total_step_lim:
					print('Total Step limit of {} Global timesteps reahed'.format(total_step_lim))
					exit()
				total_step += 1
				step += 1

				if render:
					if ep % rf == 0:
						env.render()

				action = agent.act(curr_obs)
				next_obs,reward,done,info = env.step(action)
				next_obs = next_obs.__array__(dtype=np.float32)

				if clip:
					reward = np.sign(reward)
				score += reward

				agent.remember(curr_obs,action,reward,next_obs,done)
				curr_obs = next_obs

				agent.step_update(total_step)

				if done:
					break
			
			if os.path.isfile(agent.state_save_path):
				os.remove(agent.state_save_path)
			np.savez(agent.state_save_path,ep=ep,total_step=total_step,epsilon=agent.epsilon)
			score_window.append(score)
			avg_score = np.mean(score_window)
			print('Episode {} | Global Timestep {}'.format(ep,total_step))
			print('Score: {} | Last {} episodes Avg score: {}'.format(score,window_len,avg_score))

	def get_agent(self,game_name,mode,input_dims,action_space):
		if mode == 'test':
			return DDQNPlayer(game_name,input_dims,action_space)
		elif mode == 'train':

			mem_size = self.args.buffer_size
			gamma = self.args.gamma
			batch_size = self.args.batch_size
			alpha = self.args.learning_rate
			save_freq = self.args.save_freq
			target_train_freq = self.args.target_train_freq
			replay_start_size = self.args.replay_start
			train_freq = self.args.train_freq
			collab = self.collab

			return DDQNLearner(game_name,input_dims,action_space,mem_size,gamma,batch_size,alpha,save_freq,target_train_freq,replay_start_size,train_freq,collab)


if __name__ == '__main__':
	AtariRL()


