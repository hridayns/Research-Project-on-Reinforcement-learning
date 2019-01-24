import os

class DataPaths:
	def __init__(self,env_name,drive=False):
		self.drive = drive
		
		local_root_path_components = [os.getcwd(),'data',env_name]
		drive_root_path_components = ['content','drive','My Drive','data',env_name]

		self.root_path = os.path.join(*local_root_path_components)
		if self.drive:
			self.root_path = os.path.join(*drive_root_path_components)

		self.save_folders = {
			'plot': os.path.join(self.root_path,'plots'),
			'plot_data': os.path.join(self.root_path,'plot_data'),
			'model': os.path.join(self.root_path,'saved_models')
		}

		for k in self.save_folders:
			if not os.path.exists(self.save_folders[k]):
				os.makedirs(self.save_folders[k])

	def show_paths(self):
		print('Root: {}'.format(self.root_path))
		print('Plots: {}'.format(self.save_folders['plots']))
		print('Models: {}'.format(self.save_folders['models']))