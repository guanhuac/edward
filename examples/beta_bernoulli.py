#!/usr/bin/env python
"""A simple coin flipping example. Inspired by Stan's toy example.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Beta, Empirical

ed.set_seed(42)

# DATA
x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

# MODEL
p = Beta(1.0, 1.0)
x = Bernoulli(probs=p, sample_shape=10)

# INFERENCE
def posterior(x=None):
  qp = Empirical(tf.Variable(tf.zeros([1000]) + 0.5), name="p")
  return qp

def proposal(x=None):
  proposal_p = Beta(3.0, 9.0, name="proposal_p")
  return proposal_p

# TODO update? maybe just transition of the posterior chain
# `update` is a function of the realized data (tfe.Tensor) and returns
# a train operation.
# ed.automate(train_op, x_data)
update = ed.metropolis_hastings(
    model,
    posterior,
    proposal,
    latent_align=lambda name: "qp" if name == "p" else name,
    data_align=lambda name: 0 if name == "x" else name,
    proposal_align=lambda name: "proposal_p" if name =="qp" else name,
    x_data)
# update = ed.hmc(
#   model,
#   posterior,
#   data)

sess = tf.Session()  # ed.get_session() not needed
tf.global_variables_initializer().run()  # presumably not needed
for _ in range(1000):
  info_dict = update(x_data)
