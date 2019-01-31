# Internal Imports
from utils import arg_parser
from utils.AtariWrapper import GymAtari
from utils.AtariWrapperNew import make_atari,wrap_deepmind
from utils.DataPaths import DataPaths
from utils.Logger import Logger
from utils.Plotter import Plotter
from game_model.ddqn import DDQNLearner,DDQNPlayer

# External Imports
import os
import gym
import numpy as np
from collections import deque

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class AtariRL:
	def __init__(self,args):

		game_name = args.game_name
		mode = args.mode
		env_name = game_name + 'Deterministic-v4'

		data_paths = DataPaths(
			env_name=game_name,
			drive=args.drive_save
		)
		self.plotter = Plotter(
			env_name=game_name,
			plot_types=[
				'avg_scores_ep',
				'avg_scores_ts',
				'avg_scores_100_ep',
				'avg_scores_100_ts',
				'scores_ep',
				'scores_ts',
				'high_scores_ep',
				'high_scores_ts',
				'low_scores_ep',
				'low_scores_ts',
				'avg_loss_ep',
				'avg_acc_ep',
				'timesteps_ep'
			],
			interval_types=[
				'overall',
				'window'
			],
			plot_interval=args.plot_interval,
			data_paths=data_paths
		)
		self.logger = Logger(
			env_name=game_name,
			log_types=[
				'avg_score',
				'highest_score',
				'lowest_score',
				'avg_loss',
				'avg_acc'
			],
			save_interval=args.plot_interval,
			data_paths=data_paths
		)

		# env = GymAtari.wrap(gym.make(env_name),stack_size=args.stack_size,episodic_life=args.episodic_life)
		env = make_atari(env_name)
		env = wrap_deepmind(env, frame_stack=True, scale=False)
		
		input_dims = env.reset().__array__().shape[1:]

		agent = self.get_agent(game_name,mode,input_dims,env.action_space,args,data_paths=data_paths)
		self.game_loop(agent,env,args.render,args.render_freq,args.episodes,args.total_step_lim,args.clip)

	def game_loop(self,agent,env,render,rf,episodes,total_step_lim,clip):
		ts = self.logger.log_data['ts']
		while True:
			# if ep >= episodes:
			# 	print('Episode limit of {} episodes reached'.format(episodes))
			# 	exit()
			curr_obs = env.reset()
			curr_obs = curr_obs.__array__(dtype=np.uint8)

			t = 0
			score = 0
			ep_loss = 0
			ep_acc = 0
			replay_count = 0

			while True:
				if ts >= total_step_lim:
					print('Total Step limit of {} Global timesteps reached'.format(total_step_lim))
					exit()
				t += 1
				ts += 1

				if render:
					if self.logger.log_data['epoch'] % rf == 0:
						env.render()

				action = agent.act(curr_obs)
				
				# action = env.action_space.sample()
				next_obs,reward,done,info = env.step(action)
				# reward = np.int8(reward)
				next_obs = next_obs.__array__(dtype=np.uint8)

				if clip:
					reward = np.sign(reward)
				score += reward

				agent.remember(curr_obs,action,reward,next_obs,done)
				curr_obs = next_obs

				hist = agent.step_update(ts)
				if hist:
					ep_loss += hist['loss'][0]
					ep_acc += hist['acc'][0]
					replay_count += 1

				if done:
					env.close()
					break

			agent.save_params()
			if replay_count > 0:
				ep_loss /= replay_count
				ep_acc /= replay_count
			self.logger.log_state(t,score,ep_loss,ep_acc)
			self.plotter.plot_graph(self.logger.log_data)

	def get_agent(self,game_name,mode,input_dims,action_space,args,data_paths=None):
		if mode == 'test':
			return DDQNPlayer(game_name,input_dims,action_space,data_paths=data_paths)
		elif mode == 'train':

			mem_size = args.buffer_size
			gamma = args.gamma
			batch_size = args.batch_size
			alpha = args.learning_rate
			save_freq = args.save_freq
			target_train_freq = args.target_train_freq
			replay_start_size = args.replay_start
			train_freq = args.train_freq

			return DDQNLearner(game_name,input_dims,action_space,mem_size,gamma,batch_size,alpha,save_freq,target_train_freq,replay_start_size,train_freq,data_paths=data_paths)


if __name__ == '__main__':
	args = arg_parser.parse()
	AtariRL(args)


