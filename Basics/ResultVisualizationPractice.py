)

train=tf.train.GradientDescentOptimizer(0.1).minimize(Error)
init=tf.global_variables_initializer()
with tf.Session() as sess:
	dic={xInput:xData, yInput:yData}
	sess.run(init)
	for step in xrange(0,301):
		sess.run(train,feed_dict=dic)
		if(step%30 is 0):
			try:
				ax.lines.remove(lines[0])
			except Exception:
				pass
			print(sess.run(Error,feed_dict=dic))
			predictionVal=sess.run(prediction,feed_dict=dic)
			lines=ax.plot(xData,predictionVal,'r-',lw=5)
			plt.pause(0.1)
