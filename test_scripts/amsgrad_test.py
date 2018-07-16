from __future__ import print_function

import tensorflow as tf

from tensorflow.python.training.amsgrad import *
import matplotlib.pyplot as plt

def plot_results(y_values, names, y_label):
  for m in range(len(names)):
    plt.plot(range(len(y_values[m])), y_values[m], label=names[m])
  plt.ylabel(y_label)
  plt.legend()
  plt.show()

LR = 1e-1
N_STEPS = 5000
REPORT_STEPS = 100
RAND_SEED = 100


class Toy_model():
    def __init__(self, name, seed=None):
        self.name = name
        with tf.name_scope('Toy_model'):
            self.a = tf.placeholder(tf.float32)
            self.x = tf.Variable(tf.random_uniform([1], minval=-1., 
                                                   maxval=1., seed=seed))
            self.loss = self.a * self.x


n_models  = 2
names = ['ADAM', 'AMS']
models = [Toy_model(names[m], RAND_SEED) for m in range(n_models)]
train_steps = [tf.train.AdamOptimizer(LR).minimize(models[0].loss),
               AMSGradOptimizer(LR).minimize(models[1].loss)]
#You can also try optimizers.Adam(LR)

x_vals = [[] for m in range(n_models)]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(N_STEPS):
        a_val = 1010 if step % 101 == 1 else -10
        for m in range(n_models):
            sess.run(train_steps[m], feed_dict={models[m].a: a_val})
            if step % REPORT_STEPS == 0:
                x_val = sess.run(models[m].x, feed_dict={models[m].a: a_val})
                print('step %d, %s, x: %.3g' % (step, models[m].name, x_val))
                x_vals[m].append(x_val[0])
plot_results(x_vals, names, 'x_val')