from __future__ import division, print_function, absolute_import
import tensorflow as tf


"""
    We need to be able to modify the graph to get new outputs with the same input. Variables allow us to add trainable parameters to a graph.
    Constants are initialized when you call tf.constant, and their value can never change. By contrast, variables are not initialized when you call tf.Variable.
"""

#Model Parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
#Model Input and Output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W*x + b

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
#training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
file_writer = tf.summary.FileWriter('logs', sess.graph)
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

#evaluate training accurancy
cur_w, cur_b, cur_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(cur_w, cur_b, cur_loss))