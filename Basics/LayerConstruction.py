import tensorflow as tf
import numpy as np

def addLayer(inputs, inSize, outSize, act=None):
	weights=tf.Variable(tf.random_normal([inSize,outSize]))
	biases=tf.Variable(tf.zeros([1,outSize])+0.1)
	linearOut=tf.matmul(inputs,weights)+biases
	if(act==None):
		return linearOut
	else:
		return act(linearOut)
