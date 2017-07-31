from numpy import *
import tensorflow as tf
import matplotlib.pyplot as plt

def addLayer(inputs, inSize, outSize, act=None):
	weights = tf.Variable(tf.random_normal([inSize,outSize]))
	biases = tf.Variable(tf.random_normal([1,outSize]))
	linearOut = tf.matmul(inputs,weights)+biases
	if act==None: return linearOut
	else: return act(linearOut)


x = linspace(-10,10,6000)[:,newaxis]
y = 5*sin(2*pi*x)

xInput = tf.placeholder(tf.float32,[None,1])
yInput = tf.placeholder(tf.float32,[None,1])

lSize = [1,100,100,1]
acts = [tf.sin, tf.tanh, None]
n = len(acts)
prediction = xInput
for i in xrange(n):
	prediction = addLayer(prediction,lSize[i],lSize[i+1],acts[i])

Error = tf.reduce_mean(
	tf.reduce_sum(
		tf.square(yInput-prediction),
		reduction_indices=[1]
	)
)


fig = plt.figure()
fig.show()
train=tf.train.GradientDescentOptimizer(0.01).minimize(Error)
init=tf.global_variables_initializer()
with tf.Session() as sess:
	dic = {xInput:x, yInput:y}
	sess.run(init)
	for step in xrange(3000):
		sess.run(train,feed_dict=dic)
		if(step%30==0):
			fig.clear()
			plt.figure(fig.number)
			print(sess.run(Error,feed_dict=dic))
			pred = sess.run(prediction,feed_dict=dic)
			plt.plot(x,pred)
			plt.plot(x,y)
			plt.pause(0.1)


