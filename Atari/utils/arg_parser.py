import argparse
import sys

def parse():

	parser = argparse.ArgumentParser(description='Parse parameters.')
	parser.add_argument('-game','--game_name',help='Name of environment. Default is \'Breakout\'',default='Breakout')
	parser.add_argument('-m','--mode',help='Available modes: train, test. Default is \'train\'',default='train')
	parser.add_argument('--render',help='Set this flag to render the environment',action='store_true')
	# if '--render' in sys.argv:
	parser.add_argument('-rfq','--render_freq',help='Render environment every N episodes',default=10)
	parser.add_argument('-eps','--episodes',help='Number of episodes',type=int,default=5000)
	parser.add_argument('-tsl','--total_step_lim',help='Limit on total number of global timesteps',type=int,default=5000000)
	parser.add_argument('-mem','--buffer_size',help='Replay Buffer size',type=int,default=50000)
	parser.add_argument('-g','--gamma',help='Discount factor',type=float,default=0.99)
	parser.add_argument('-b','--batch_size',help='Batch size',type=int,default=32)
	parser.add_argument('-lr','--learning_rate',help='Learning rate',type=float,default=0.00025)
	parser.add_argument('--save_freq',help='Model saving frequency (in timesteps)',type=int,default=10000)
	parser.add_argument('--target_train_freq',help='Target model train frequency(in timesteps)',type=int,default=40000)
	parser.add_argument('--replay_start',help='Global timestep after which replay can start',type=int,default=50000)
	parser.add_argument('-c','--clip',help='Clip rewards to scale?',action='store_true')

	args = parser.parse_args()
	return args