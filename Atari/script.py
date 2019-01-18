# Internal Imports
from utils import arg_parser
from utils.AtariWrapper import GymAtari
from game_model.ddqn import DDQNLearner,DDQNPlayer

# External Imports
import gym
import numpy as np

args = arg_parser.parse()

class AtariRL:
	def __init__(self):
		game_name = args.game_name
		mode = args.mode
		env_name = game_name + 'Deterministic-v4'
		env = GymAtari.wrap(gym.make(env_name),4)
		input_dims = env.reset().__array__().shape[1:]

		agent = self.get_agent(game_name,mode,input_dims,env.action_space)
		self.game_loop(agent,env,args.render,args.render_freq,args.episodes,args.total_step_lim,args.clip)

	def game_loop(self,agent,env,render,rf,episodes,total_step_lim,clip):
		
		ep = 0
		total_step = 0
		
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

				reward = np.uint8(reward)
				if clip:
					reward = np.sign(reward,dtype=np.uint8)
				score += reward

				agent.remember(curr_obs,action,reward,next_obs,done)
				curr_obs = next_obs

				agent.step_update(total_step)

				if done:
					break
			print('Episode {} ---> Score: {}'.format(ep,score))

	def get_agent(self,game_name,mode,input_dims,action_space):
		if mode == 'test':
			return DDQNPlayer(game_name,input_dims,action_space)
		elif mode == 'train':
			return DDQNLearner(game_name,input_dims,action_space)


if __name__ == '__main__':
	AtariRL()


