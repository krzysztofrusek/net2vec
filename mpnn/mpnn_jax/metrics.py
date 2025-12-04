import jax.numpy as jnp

def r2(y, yhat):
    return 1 - jnp.sum((y - yhat)**2) / jnp.sum((y - y.mean())**2)

def mae(y, yhat):
    return jnp.mean(jnp.abs(y - yhat))

def mse(y, yhat):
    return jnp.mean(jnp.square(y - yhat))

def pearson(y, yhat):
    return jnp.corrcoef(y, yhat)[0, 1]

