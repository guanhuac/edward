"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward.inferences.bigan_inference import *
from edward.inferences.conjugacy import *
from edward.inferences.gan_inference import *
# from edward.inferences.gibbs import *
# from edward.inferences.hmc import *
from edward.inferences.implicit_klqp import *
from edward.inferences.inference import *
from edward.inferences.klpq import *
from edward.inferences.klqp import *
from edward.inferences.laplace import *
from edward.inferences.map import *
# from edward.inferences.metropolis_hastings import *
# from edward.inferences.monte_carlo import *
# from edward.inferences.sgld import *
# from edward.inferences.sghmc import *
from edward.inferences.wake_sleep import *
from edward.inferences.wgan_inference import *

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    'bigan_inference',
    'complete_conditional',
    'gan_inference',
    'implicit_klqp',
    'Gibbs',
    'HMC',
    'klpq',
    'klqp',
    'klqp_reparameterization',
    'klqp_reparameterization_kl',
    'klqp_score',
    'klqp_score_rb',
    'laplace',
    'map',
    'MetropolisHastings',
    'MonteCarlo',
    'SGLD',
    'SGHMC',
    'wake_sleep',
    'wgan_inference',
]

remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)
