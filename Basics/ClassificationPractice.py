import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

def addLayer(
	inputs,
	inSize,
	outSize,
	act=None
):
	wMat=tf.Variable(tf.random_normal([inSize,outSize]))
	bMat=tf.Variable(tf.zeros([1,outSize])+0.1)
	linearOut=tf.matmul(inputs,wMat)+bMat
	if(act==None):
		return linearOut
	else:
		return act(linearOut)

xInput=tf.placeholder(tf.float32,[None,28*28])
yInput=tf.placeholder(tf.float32,[None,10])

prediction=addLayer(xInput,28*28,10,act=tf.nn.softmax)

error=tf.reduce_mean(
	-tf.reduce_sum(
		yInput*tf.log(prediction),
		reduction_indices=[1]
	)
)
train=tf.train.GradientDescentOptimizer(0.5).minimize(error)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	def accuracy(vxInput, vyInput):
		global prediction
		vDic={xInput: vxInput}
		vPrediction=sess.run(prediction,feed_dict=vDic)
		isCorrect=np.equal(
			np.argmax(vPrediction,1),
			np.argmax(vyInput,1)
		)
		return np.mean(
			isCorrect.astype(np.float32)
		)
	for cnt in xrange(1000):
		batchX,batchY=mnist.train.next_batch(100)
		dic={xInput:batchX, yInput:batchY}
		sess.run(train,feed_dict=dic)
		if(cnt%50 is 0):
			print(
				accuracy(
					mnist.test.images,
					mnist.test.labels
				)
			)	
