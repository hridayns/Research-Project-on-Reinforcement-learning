import gym
import os
import numpy as np
from time import sleep

from config import get_paths
from models import DDQNLearner,DDQNPlayer
from utils import make_atari,wrap_deepmind,parse_args
from utils import Logger,Plotter

args = parse_args()
# for arg in vars(args):
# 	print(arg, getattr(args, arg))

ENV_NAME = args.env_name
ENV_VER = args.env_version
ENV_GYM = ENV_NAME + ENV_VER

save_dirs = get_paths(
	drive=args.drive_save,
	env_name=ENV_NAME
)

PRINT_FREQ_EP = args.log_freq
SAVE_MODEL_FREQ = args.save_freq
LEARNING_START = args.learn_start

logger = Logger(save_dirs=save_dirs,log_types=[],log_freq=args.log_freq,mode=args.mode)
plotter = Plotter(
	save_dirs=save_dirs,
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
	plot_freq=args.plot_freq,
	mode=args.mode
)

env = make_atari(ENV_GYM)
env = wrap_deepmind(env, frame_stack=True, scale=False)

if args.mode == 'train':
	agent = DDQNLearner(
		env=env,save_dirs=save_dirs,save_freq=args.save_freq,gamma=args.gamma,batch_size=args.batch_size,learning_rate=args.learning_rate,buffer_size=args.buffer_size,learn_start=args.learn_start,target_network_update_freq=args.target_network_update_freq,train_freq=args.train_freq,tot_steps=args.total_step_lim
	)
else:
	agent = DDQNPlayer(
		env=env,save_dirs=save_dirs,learning_rate=args.learning_rate,epsilon_test=args.epsilon_test
	)


saved_mean_reward = None
obs = env.reset().__array__()

reset = True
score = 0
ep_steps = 0
learn_count = 0
ep_loss = 0
ep_acc = 0
for t in range(args.total_step_lim):

	if args.render:
		if args.mode == 'test':
			env.render()
			sleep(0.01)
		else:
			if logger.data['episode'] % args.render_freq == 0:
				env.render()
				sleep(0.01)

	action = agent.act(obs)#env.action_space.sample()
	agent.update_exploration(t)
	reset = False
	new_obs, rew, done, _ = env.step(action)
	new_obs = new_obs.__array__()
	agent.remember(obs,action,rew,new_obs,done)
	obs = new_obs

	score += rew
	ep_steps += 1

	if done:
		obs = env.reset()
		if args.render:
			env.close()

		if learn_count > 0:
			ep_loss /= learn_count
			ep_acc /= learn_count

		logger.update_state(ep_steps,score,ep_loss,ep_acc)
		plotter.plot_graph(logger.data)
		score = 0
		ep_steps = 0
		reset = True


	hist = agent.step_update(t)
	if hist:
		ep_loss += hist['loss'][0]
		ep_acc += hist['acc'][0]
		learn_count += 1

	if args.mode == 'test':
		logger.log_state(done,args.epsilon_test)
	else:
		logger.log_state(done,agent.exploration.value(t))
	agent.save(t,logger)
