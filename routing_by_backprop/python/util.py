import contextlib
import itertools as it
import os
import sys
import time
from typing import NamedTuple, Any, Callable, Tuple, List

import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
from absl import flags


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)


class WelfordState(NamedTuple):
    k: int
    m: Any
    s: Any

class MeanStdReducer(NamedTuple):
    init: Callable
    update: Callable
    stats: Callable

class Stats(NamedTuple):
    mean:Any
    std: Any

def _stats_fn(state: WelfordState):
    return Stats(
        mean=state.m,
        std=jax.tree_map(lambda x: jnp.sqrt(x/(state.k-1)), state.s)
    )

def welford()->MeanStdReducer:
    '''
    https://www.johndcook.com/blog/standard_deviation/
    :return:
    '''
    def init_fn(x0:Any):
        return WelfordState(
            k=jnp.ones(()),
            m=x0,
            s=jax.tree_map(jnp.zeros_like, x0)
        )
    @jax.jit
    def update_fn(state:Any, x:Any):
        k = state.k+jnp.ones(())
        m = state.m + (x-state.m)/k
        s = state.s + (x-state.m)*(x-m)
        return WelfordState(
            k=k,
            m=m,
            s=s
        )
    return MeanStdReducer(init_fn, update_fn, _stats_fn)


def batched_welford()->MeanStdReducer:
    w = welford()
    @jax.jit
    def update_fn(state:Any, x:Any):
        state = jax.lax.scan(
            lambda *arg: (w.update(*arg), None),
            state,
            x
        )[0]
        return state

    return MeanStdReducer(w.init, update_fn, _stats_fn)

def parser(args):
    """Tries to parse the flags, print usage, and exit if unparseable.
    Args:
      args: [str], a non-empty list of the command line arguments including
          program name.
    Returns:
      [str], a non-empty list of remaining command line arguments after parsing
      flags, including program name.
    """
    try:
        return flags.FLAGS(args, known_only=True)
    except flags.Error as error:
        sys.stderr.write('FATAL Flags parsing error: %s\n' % error)
        sys.stderr.write('Pass --helpshort or --helpfull to see help on flags.\n')
        sys.exit(1)

class MultiSGDState(NamedTuple):
    #opts:list
    states:tuple
    #fun:Callable
    #vfun: Callable
    # theta:Any


def multi_sgd(learning_rates: List[float],fun:Callable)-> Tuple[Callable, Callable]:
    opts = [optax.sgd(learning_rate=lr) for lr in learning_rates]

    def init(theta:Any):
        opt_states = tuple([o.init(theta) for o in opts])
        return MultiSGDState(
            states=opt_states
        )

    vfun = jax.vmap(fun)
    gfun = jax.grad(fun)

    def update(theta:Any, state:MultiSGDState)->Any:
        proposals = [optax.apply_updates(theta, o.update(gfun(theta), s)[0]) for o, s in zip(opts, state.states)]
        dense_poposals = jax.tree_multimap(lambda *xs: jnp.stack(xs), *proposals)

        values = vfun(dense_poposals)
        min_idx = jnp.argmin(values)

        theta = jax.tree_map(lambda a: a[min_idx, ...], dense_poposals)
        return theta

    return init, update

class tlist(list):
    def __init__(self):
        super().__init__([time.time_ns()])
    def diff(self):
        return self[-1]-self[0]

@contextlib.contextmanager
def timer():
    begin = tlist()
    try:
        yield begin
    finally:
        begin.append(time.time_ns())


class TensorBoard:
    def __init__(self, logdir):
        self.logdir = logdir
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.writer = tf.summary.create_file_writer(self.logdir)

    def scalar(self, name, data, step):
        with self.writer.as_default():
            tf.summary.scalar(name, data, step)