import argparse
import sys

def parse():

	parser = argparse.ArgumentParser(description='Parse parameters.')
	parser.add_argument('--test',help='Set this flag to test the agent',action='store_true')
	parser.add_argument('--render',help='Set this flag to render the environment',action='store_true')
	if '--render' in sys.argv:
		parser.add_argument('--render_freq',help='Render environment every N episodes',default=10)
	parser.add_argument('--episodes',help='Number of episodes',type=int)
	parser.add_argument('--buffer_size',help='Replay Buffer size',type=int)
	parser.add_argument('--gamma',help='number of episodes',type=float)
	parser.add_argument('--batch_size',help='number of episodes',type=int)
	parser.add_argument('--lr',help='Learning rate',type=float)
	parser.add_argument('--save_freq',help='Model saving frequency',type=int)

	args = parser.parse_args()
	return args