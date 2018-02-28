import numpy as np
import tensorflow as tf

x_data = np.random.rand(100).astype(np.float32)
y_data = 0.5*x_data+0.1

m = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = m*x_data+b

#minimizar errores de media cuadratica

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(350):
	sess.run(train)
	if i % 20 == 0:
		print(i, sess.run(m), sess.run(b))
