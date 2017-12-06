from __future__ import division, print_function, absolute_import
import tensorflow as tf


"""
    TensorFlow constants, it takes no inputs, and it outputs a value it stores internally
""" 
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
sess = tf.Session()
file_writer = tf.summary.FileWriter('logs', sess.graph)
print(sess.run(node3))