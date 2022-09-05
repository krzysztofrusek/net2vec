import jax
import jax.numpy as jnp
from typing import Callable

def sumsoftmax(temperature:float=1.)->Callable:
    '''
    Differentiable approximation to `maximum` function
    :param temperature: sharpenss parameter, as t goes to 0 this function goes to perfect maximum function
    :return: Cost function
    '''
    @jax.jit
    def sumsoftmax_fn(x:jnp.ndarray)->float:
        scaled_x = x/temperature
        return temperature*jnp.dot(jax.nn.softmax(scaled_x),scaled_x)
    return  sumsoftmax_fn

def average_queue(b:int=4)->Callable:
    '''
    Makes cost function based on average queue lenght in M/M/1/b systen
    https://www.wolframalpha.com/input/?i=plot+%28sum+n+x%5En%2C+n%3D0+to+30%29%2F%28sum+x%5En%2C+n%3D0+to+30+%29+x%3D0+to+2

    :param b: buffer
    :return: Cost function
    '''
    @jax.vmap
    def average_fn(x:jnp.ndarray)->jnp.ndarray:
        states = jnp.arange(0,b+1)
        logits = jnp.repeat(jnp.log(x),b+1)*states
        pi = jax.nn.softmax(logits)
        average = jnp.dot(pi,states)
        return average

    @jax.jit
    def cost_fn(x:jnp.array)->float:
        return jnp.sum(average_fn(x))

    return cost_fn