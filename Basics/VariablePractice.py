# Concept:
# A tensoflow variable is not only a value container,
# it is also a structure that can be connected to
# other variable in a graph structure

# Create a Variable
state=tf.Variable(0, name='counter')

# Create a constant
# Constant is a unchangeable variable
# note that any arithmatics between
# Variables and Constants shall give out
# a Variable dependent on its generators
one=tf.constant(1)
newVal=state+one

# Create an instruction
# An instruction is a process
# that can be carried out by a process
update = tf.assign(state,newVal)

# Create an initializer
# An initializer is an instruction that
# initialize all variables
init=tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for cnt in xrange(5):
		sess.run(update)
		print(sess.run(state))
