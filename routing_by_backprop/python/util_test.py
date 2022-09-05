import unittest

import jax

import util
from util import *
from functools import reduce
import numpy as np

class ReducerTestCase(unittest.TestCase):
    def test_meanstd(self):
        x=jnp.array([1,2,3,4,5.,6,7])
        m=jnp.mean(x)
        s=jnp.std(x, ddof=1)
        mean_std = welford()

        state = mean_std.init(x[0])
        state = reduce(mean_std.update, x[1:], state)
        stats = mean_std.stats(state)

        self.assertTrue(jnp.allclose(m, stats.mean))
        self.assertTrue(jnp.allclose(s, stats.std))

    def test_lax_scan(self):
        x=jnp.array([1,2,3,4,5.])
        true_stats=Stats(mean=jnp.mean(x), std=jnp.std(x, ddof=1))

        mean_std = welford()

        scan_stats = mean_std.stats(
            jax.lax.scan(
                lambda *arg: (mean_std.update(*arg), None),
                mean_std.init(x[0]),
                x[1:]
            )[0]
        )

        for t, l in zip(true_stats, scan_stats):
            self.assertTrue(np.allclose(t,l))

    def test_batched(self):
        x=[np.array([1,2,3.]),
           np.array([4,5.]),
           np.array([6.])]
        wb = batched_welford()

        ZERO = np.array([0.])
        state = wb.init(ZERO)
        state = reduce(wb.update, x, state)
        stats = wb.stats(state)

        xx = np.concatenate([ZERO]+x, axis=0)
        true_stats = Stats(mean=np.mean(xx), std=np.std(xx, ddof=1))
        for t, l in zip(true_stats, stats):
            self.assertTrue(np.allclose(t,l))



class MultiSGDTest(unittest.TestCase):

    def test_lm(self):
        x = np.random.normal(size=100)
        y = 2 * x + 1 + np.random.normal(scale=0.1, size=100)

        def loss(theta):
            a, b = theta
            return jnp.mean(jnp.square(y - a * x - b))


        init, update = multi_sgd(np.logspace(-5,1,10),fun=loss)
        update = jax.jit(update)
        theta = (1., 0.)
        state = init(theta = (1.,0.))

        for _ in range(50):
            theta=update(theta,state)

        self.assertTrue(np.allclose(theta,(2,1),rtol=0.01))


class TimerTest(unittest.TestCase):
    def test_timer(self):
        i=4
        with util.timer() as pt:
            i=2
        self.assertTrue(pt.diff())

if __name__ == '__main__':
    unittest.main()
