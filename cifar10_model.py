import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,Sequential,optimizers,layers,metrics

def preprocess(x,y):
	x = 2 * tf.cast(x,dtype=tf.float32) / 255. - 1
	y = tf.cast(y,dtype=tf.int32)
	return x,y


class MyDense(layers.Layer):
	
	def __init__(self,inp_dim,outp_dim):
		super(MyDense,self).__init__()

		self.kernel = self.add_variable('w',[inp_dim,outp_dim])
		# self.bias = self.add_variable('b',[outp_dim])

	def call(self,inputs,training=None):
		out = inputs @ self.kernel
		
		return out


class MyNetwork(keras.Model):
 	
 	def __init__(self):
 		super(MyNetwork, self).__init__()
 		
 		self.fc1 = MyDense(32*32*3,256)
 		self.fc2 = MyDense(256,128)
 		self.fc3 = MyDense(128,64)
 		self.fc4 = MyDense(64,32)
 		self.fc5 = MyDense(32,10)
 	
 	def call(self,inputs,training=None):
 		logits = tf.reshape(inputs,[-1,32*32*3])

 		logits = self.fc1(logits)
 		logits = tf.nn.relu(logits)
 		logits = self.fc2(logits)
 		logits = tf.nn.relu(logits)
 		logits = self.fc3(logits)
 		logits = tf.nn.relu(logits)
 		logits = self.fc4(logits)
 		logits = tf.nn.relu(logits)
 		logits = self.fc5(logits)

 		return logits

		
if __name__ == '__main__':
	(x,y),(x_test,y_test) = datasets.cifar10.load_data()
	# print(y.shape,y_test.shape)
	y = tf.squeeze(y)
	y_test = tf.squeeze(y_test)
	y = tf.one_hot(y,depth=10)
	y_test = tf.one_hot(y_test,depth=10)
	# 此时x仍然是numpy类型(自己备注的)
	print('datasets',x_test.shape,y_test.shape,x.shape,y.shape,x.min(),x.max())

	batchsz=128

	train_db = tf.data.Dataset.from_tensor_slices((x,y))
	train_db = train_db.map(preprocess).shuffle(50000).batch(batchsz)
	test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
	test_db = test_db.map(preprocess).batch(batchsz)

	sample = next(iter(train_db))
	print('batch:',sample[0].shape,sample[1].shape)

	network = MyNetwork()
	network.compile(optimizer=optimizers.Adam(lr=1e-3),
		loss=tf.losses.CategoricalCrossentropy(from_logits=True),
		metrics=['accuracy'])
	network.fit(train_db,epochs=5,validation_data=test_db,validation_freq=1)

	network.evaluate(test_db)
	network.save_weights('ckpt/weights.ckpt')
	# del network
	# print('saved to ckpt/weights.ckpt')

	# network = MyNetwork()
	# network.compile(optimizer=optimizers.Adam(lr=1e-3),
	# 	loss=tf.losses.CategoricalCrossentropy(from_logits=True),
	# 	metrics=['accuracy'])
	# network.load_weights('ckpt/weights.ckpt')
	# print('loaded weights from file')
	# network.evaluate(test_db)

