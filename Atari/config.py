import os

def get_paths(drive=False,env_name=None):
	if drive:
		root = '/content/drive/My Drive'
	else:
		root = os.getcwd()

	root_path_components = [root,'data',env_name]
	root_path = os.path.join(*root_path_components)

	save_dirs = {
		'plots': None,
		'checkpoints': None,
		'plot_data': None
	}

	for k in save_dirs:
		save_dirs[k] = os.path.join(root_path,k)
		if not os.path.exists(save_dirs[k]):
				os.makedirs(save_dirs[k])

	return save_dirs