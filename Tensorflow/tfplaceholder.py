from __future__ import division, print_function, absolute_import
import tensorflow as tf

"""
    A graph can be parameterized to accept external inputs, known as placeholders.
    A placeholder is a promise to provide a value later.
    The preceding three lines are a bit like a function or a lambda in which we define two input parameters (a and b) and then an operation on them.
"""
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
add_and_triple = adder_node * 3
sess = tf.Session()
file_writer = tf.summary.FileWriter('logs', sess.graph)
# print(sess.run(adder_node, {a: 3, b: 4.5}))
# print(sess.run(adder_node, {a: [1,2], b:[3,4]}))
print(sess.run(add_and_triple, {a: 4.2, b: 6.8}))