import tensorflow as tf
import numpy as np

input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)
output=input1**input2

with tf.Session() as sess:
        dic={input1: 3, input2: 4}
        print(sess.run(output,feed_dict=dic))
