	else:
		return act(linearOut)

# Create Test Case Input
xData=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,xData.shape)
yData=np.square(xData)-0.5

# Structure
xInput=tf.placeholder(tf.float32,[None,1])
yInput=tf.placeholder(tf.float32,[None,1])

# Add a hidden layer and output layer
layer1=addLayer(xInput,1,10,act=tf.nn.relu)
prediction=addLayer(layer1,10,1,act=None)

Error=tf.reduce_mean(
	tf.reduce_sum(
		tf.square(yInput-prediction),
		reduction_indices=[1]
	)
)

train=tf.train.GradientDescentOptimizer(0.1).minimize(Error)
init=tf.global_variables_initializer()
with tf.Session() as sess:
	dic={xInput:xData, yInput:yData}
	sess.run(init)
	for step in xrange(0,301):
		sess.run(train,feed_dict=dic)
		if(step%30 is 0):
			print(sess.run(Error,feed_dict=dic))
