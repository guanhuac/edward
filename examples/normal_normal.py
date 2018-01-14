#!/usr/bin/env python
"""Normal-normal model using Hamiltonian Monte Carlo."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Empirical, Normal
from edward.util import get_session, Progbar

ed.set_seed(42)

def model():
  """Normal-Normal with known variance."""
  mu = Normal(loc=0.0, scale=1.0, name="mu")
  x = Normal(loc=mu, scale=1.0, sample_shape=50, name="x")
  return x

def variational():
  qmu = Normal(loc=tf.get_variable("loc", []),
               scale=tf.nn.softplus(tf.get_variable("shape", [])),
               name="qmu")
  return qmu

variational = tf.make_template("variational", variational)

x_data = np.array([0.0] * 50)

# analytic solution: N(loc=0.0, scale=\sqrt{1/51}=0.140)
loss = ed.klqp_reparameterization(
    model,
    variational,
    align_latent=lambda name: 'qmu' if name == 'mu' else name,
    align_data=lambda name: 'x_data' if name == 'x' else name,
    x_data=x_data)

var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
grads = tf.gradients(loss, var_list)
grads_and_vars = list(zip(grads, var_list))
train_op = tf.train.AdamOptimizer().apply_gradients(grads_and_vars)
progbar = Progbar(1000)
sess = get_session()
tf.global_variables_initializer().run()
for t in range(1, 1001):
  loss_val, _ = sess.run([loss, train_op])
  if t % 50 == 0:
    progbar.update(t, {"Loss": loss_val})

# CRITICISM
qmu = variational()  # TODO why is this uninitialized?
mean, stddev = sess.run([qmu.mean(), qmu.stddev()])
print("Inferred posterior mean:")
print(mean)
print("Inferred posterior stddev:")
print(stddev)

# Check convergence with visual diagnostics.
# samples = sess.run(qmu.params)

# # Plot histogram.
# plt.hist(samples, bins='auto')
# plt.show()

# # Trace plot.
# plt.plot(samples)
# plt.show()
