import tensorflow as tf
import numpy as np

mat1=tf.constant([[3,3]])
mat2=tf.constant([
	[2],
	[2]
])
product=tf.matmul(mat1,mat2)

# Method 1
sess1=tf.Session()
result1=sess1.run(product)
print(result1)
sess1.close()

# Method 2
with tf.Session() as sess2:
	result2=sess2.run(product)
	print(result2)
