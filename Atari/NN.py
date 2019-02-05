import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten
from tensorflow.python.keras.optimizers import Adam

class NeuralNet:
	def __init__(self,input_shape,num_actions,learning_rate=0.0001,blueprint=None):
		self.model = Sequential()
		
		for i in range(blueprint['conv_layers']):
			if i == 0:
				self.model.add(Conv2D(
					input_shape=input_shape,
					filters=blueprint['filters'][i],
					kernel_size=blueprint['kernel_sizes'][i],
					strides=blueprint['strides'][i],
					padding=blueprint['paddings'][i],
					activation=blueprint['activations'][i]
				))
			else:
				self.model.add(Conv2D(
					filters=blueprint['filters'][i],
					kernel_size=blueprint['kernel_sizes'][i],
					strides=blueprint['strides'][i],
					padding=blueprint['paddings'][i],
					activation=blueprint['activations'][i]
				))

		'''
		self.model.add(Conv2D(
				input_shape=input_shape,
				filters=32,
				kernel_size=(8,8),
				strides=(4,4),
				padding='valid',
				activation='relu'
			))

		self.model.add(Conv2D(
				filters=64,
				kernel_size=(4,4),
				strides=(2,2),
				padding='valid',
				activation='relu'
			))

		self.model.add(Conv2D(
				filters=64,
				kernel_size=(3,3),
				strides=(1,1),
				padding='valid',
				activation='relu'
			))
		'''
		self.model.add(Flatten())

		self.model.add(Dense(
			units=blueprint['dense_units'],
			activation=blueprint['dense_activation']
		))

		# self.model.add(Dense(
		# 		units=512,
		# 		activation='relu'
		# 	))

		self.model.add(Dense(
			units=num_actions
		))

		self.model.compile(
			loss=self.huber_loss,
			optimizer=Adam(
				lr=learning_rate
				# rho=0.95,
				# epsilon=0.01
			),
			metrics=['acc']
		)

	def huber_loss(self,y_true, y_pred):
		return tf.losses.huber_loss(y_true,y_pred)