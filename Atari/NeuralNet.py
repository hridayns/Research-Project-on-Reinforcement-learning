from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten
from tensorflow.python.keras.optimizers import RMSprop,Adam
import tensorflow as tf

class NeuralNet:
	def __init__(self,input_dims,action_space,learning_rate=0.00025):
		self.model = Sequential()
		self.model.add(Conv2D(
				input_shape=input_dims,
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

		self.model.add(Flatten())

		self.model.add(Dense(
				units=512,
				activation='relu'
			))

		self.model.add(Dense(
			units=action_space.n
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

		# self.model.summary()

		# return self.model