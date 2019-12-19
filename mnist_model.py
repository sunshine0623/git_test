#!/usr/bin/env python
# -*- coding: utf-8 -*-
# filename: fashion_mnist.py
# author: luxiali
# datetime: 2019/10/15 21:13

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics

import io
import datetime
from matplotlib import pyplot as plt

(x,y),(x_test,y_test) = datasets.mnist.load_data()
print(x.shape,y.shape)


def preprocess(x,y):
	x = tf.cast(x,dtype=tf.float32) / 255.
	y = tf.cast(y,dtype=tf.int32)

	return x,y


def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  
  return image


def image_grid(images):
  """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
  # Create a figure to contain the plot.
  figure = plt.figure(figsize=(10,10))
  for i in range(25):
    # Start next subplot.
    plt.subplot(5, 5, i + 1, title='name')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i], cmap=plt.cm.binary)
  
  return figure


batch_size = 128

db = tf.data.Dataset.from_tensor_slices((x,y))
db = db.map(preprocess).shuffle(60000).batch(batch_size).repeat(10)

db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.map(preprocess).batch(batch_size,drop_remainder=True)

db_iter = iter(db)
sample = next(db_iter)
print(sample[0].shape,sample[1].shape)


model = Sequential([
	layers.Dense(256,activation=tf.nn.relu),
	layers.Dense(128,activation=tf.nn.relu),
	layers.Dense(64,activation=tf.nn.relu),
	layers.Dense(32,activation=tf.nn.relu),
	layers.Dense(10)
])

model.build(input_shape=[None,28*28])
model.summary()

# w = w - lr*grad
optimizer = optimizers.Adam(lr=1e-3)

current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

acc_meter = metrics.Accuracy()
loss_ce_meter = metrics.Mean()


def main():
	for step,(x,y) in enumerate(db):
		# x:[b,28,28]
		# y:[b]
		x = tf.reshape(x,[-1,28*28])

		with tf.GradientTape() as tape:
			logits = model(x)
			y_onehot = tf.one_hot(y,depth=10)
			loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot,logits))
			loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits,from_logits=True))
			loss_ce_meter.update_state(loss_ce)

		grads = tape.gradient(loss_ce,model.trainable_variables)
		optimizer.apply_gradients(zip(grads,model.trainable_variables))

		if step % 100 == 0:
			print(step,'loss',float(loss_ce),float(loss_mse))
			print(step,'loss',loss_ce_meter.result().numpy())
			loss_ce_meter.reset_states()
			with summary_writer.as_default():
				tf.summary.scalar("train-loss-ce",float(loss_ce),step=step)

		# test
		if step % 500 == 0:
			acc_meter.reset_states()
			total_correct = 0
			total_num = 0
			for x,y in db_test:
				x = tf.reshape(x, [-1, 28 * 28])
				logits = model(x)
				prob = tf.nn.softmax(logits,axis=1)
				# [b,10] → [b] pred:tf.int64,y:tf.int32
				pred = tf.argmax(prob,axis=1)
				pred = tf.cast(pred,dtype=tf.int32)
				# correct:[b],True:equal,False:not equal
				correct = tf.equal(pred,y)
				correct = tf.reduce_sum(tf.cast(correct,dtype=tf.int32))

				total_correct += int(correct)
				total_num += x.shape[0]

				acc_meter.update_state(y,pred)

			acc = total_correct / total_num
			print(step,'test acc',acc,acc_meter.result().numpy())

			# print(x.shape)
			val_images = x[:25]
			val_images = tf.reshape(val_images,[-1,28,28,1])
			with summary_writer.as_default():
				tf.summary.scalar("test-acc",float(total_correct/total_num),step=step)
				tf.summary.image("val-onebyone-images",val_images,max_outputs=25,step=step)

				val_images = tf.reshape(val_images,[-1,28,28])
				figure = image_grid(val_images)
				tf.summary.image("val-images",plot_to_image(figure),step=step)


if __name__ == '__main__':
	main()


