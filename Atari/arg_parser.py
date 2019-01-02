import argparse
import sys

def parse():

	parser = argparse.ArgumentParser(description='Parse parameters.')
	parser.add_argument('--test',help='Set this flag to test the agent',action='store_true')
	parser.add_argument('--render',help='Set this flag to render the environment',action='store_true')
	if '--render' in sys.argv:
		parser.add_argument('--render_freq',help='Render environment every N episodes',default=10)
	parser.add_argument('--episodes',help='Number of episodes')
	parser.add_argument('--buffer_size',help='Replay Buffer size')
	parser.add_argument('--gamma',help='number of episodes')
	parser.add_argument('--batch_size',help='number of episodes')
	parser.add_argument('--lr',help='Learning rate')

	args = parser.parse_args()
	return args