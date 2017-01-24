import tensorflow as tf
import numpy as np

xData=np.random.rand(100).astype(np.float32)
yData=xData*0.1+0.3

# Building Tensorflow Structure

# Create Variables
# Variable is fundamental
# It is used to predict results in NN
# It is changeable, can be optimized by various approach
Weight=tf.Variable(tf.random_uniform([1],-1,1))
Bias=tf.Variable(tf.zeros([1]))

# Build up relationships between Variables
yPredict=Weight*xData+Bias
Error=tf.reduce_mean(tf.square(yPredict-yData))

# Create optimzer and trainer
# Optimzer is a structure stands for an optimizing strategy
# Trainer is a training recipe
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(Error)

# Create initializer
# Initializer is used to activate a session
init=tf.initialize_all_variables()

# Create session
# Session is the training carier
# Can be activated by an initializer
sess=tf.Session()
sess.run(init)
# Training
# When feeded with a trainer, the session is one step further optimized
# When feeded with a variable, the session shows the content of such variable
for step in xrange(300):
        sess.run(train)
        if(step%20==0):
                print(step,sess.run(Weight),sess.run(Bias))
