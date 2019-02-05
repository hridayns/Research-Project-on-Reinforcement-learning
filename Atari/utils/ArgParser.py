import sys
import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Parse parameters.')
	parser.add_argument('-env','--env_name',help='Name of environment. Default is \'Pong\'',default='Pong')
	parser.add_argument('-env_ver','--env_version',help='Version of the environment. Default is \'NoFrameskip-v4\'',default='NoFrameskip-v4')
	parser.add_argument('-m','--mode',help='Available modes: train, test. Default is \'train\'',default='train')
	parser.add_argument('-drive','--drive_save',help='Dev argument for saving to drive',action='store_true')
	parser.add_argument('-r','--render',help='Set this flag to render the environment',action='store_true')
	if '--render' in sys.argv:
		parser.add_argument('-rfq','--render_freq',help='Render environment every N episodes',type=int,default=10)

	parser.add_argument('--save_freq',help='Saving frequency (in timesteps)',type=int,default=10000)
	parser.add_argument('--log_freq',help='Logging frequency (in episodes)',type=int,default=100)
	parser.add_argument('--plot_freq',help='Plotting frequency (in episodes)',type=int,default=100)
	parser.add_argument('--learn_start',help='Global timestep after which training can start',type=int,default=10000)

	parser.add_argument('-tsl','--total_step_lim',help='Limit on total number of global timesteps',type=int,default=10000000)
	parser.add_argument('-mem','--buffer_size',help='Replay Buffer size',type=int,default=10000)
	parser.add_argument('-g','--gamma',help='Discount factor',type=float,default=0.99)
	parser.add_argument('-b','--batch_size',help='Batch size',type=int,default=32)
	parser.add_argument('-lr','--learning_rate',help='Learning rate',type=float,default=0.0001)
	parser.add_argument('--train_freq',help='Model training frequency (in timesteps)',type=int,default=4)
	parser.add_argument('--target_network_update_freq',help='Target model train frequency(in timesteps)',type=int,default=1000)

	parser.add_argument('--epsilon_test',help='Exploration factor for test environment',type=float,default=0.02)
	# parser.add_argument('-c','--clip',help='Clip rewards to scale?',action='store_true')

	args = parser.parse_args()
	return args



