import functools

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import haiku as hk
import chex
import optax

import jax
import jax.numpy as jnp

#%%

def stair_caise_function(x:chex.Array)->chex.Array:
    p=jnp.poly(-jnp.array([-1,-0.5,0.5,1]))
    smooth = jnp.polyval(p,x)/0.2
    return jnp.ceil(smooth),smooth


#%%

x = np.linspace(-1.2,1.2,100)
y,ysmooth = stair_caise_function(x)

sns.lineplot(x=x,y=y)
plt.show()


#%%
xtrain = np.random.uniform(low=x[0],high=x[-1],size=(128,1))
ytrain,_ = stair_caise_function(xtrain)

#%% nn

@hk.transform
def nn(x):
    return hk.nets.MLP([16,1],activation=jax.nn.tanh)(x)

#%%

rng = jax.random.PRNGKey(42)

params = nn.init(rng,xtrain)
opt = optax.adam(0.01)

opt_state = opt.init(params)

def loss(params,x,y):
    yhat = nn.apply(params, None, x)
    return jnp.mean(optax.l2_loss(yhat,y))

grad_loss = jax.jit(jax.grad(loss))

@jax.jit
def step(params, opt_state, x, y):
    grads = jax.grad(loss)(params, xtrain, ytrain)
    updates, opt_state = opt.update(grads,opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


for _ in range(1000):
    params, opt_state = step(params, opt_state,xtrain,ytrain)


# %%
@jax.jit
def grad_fn(params, rng, x):
    @functools.partial(jax.vmap, in_axes=(0, None, None))
    @jax.grad
    def fn(x, params, rng):
        return nn.apply(params, rng, jnp.expand_dims(x, 0))[0]

    return fn(x, params, rng)
#%%

yhat = nn.apply(params, rng, x[...,np.newaxis])[:,0]
grads = grad_fn(params,rng,x)

sns.set(style='whitegrid',
        context='paper',
        rc={ 'font.family': 'serif'}
        )

# ax = f.add_subplot(111)
fig, ax = plt.subplots(figsize=(3.5,2.16), constrained_layout=True)

sns.lineplot(x=x,y=y,label='$f(x)$', ax=ax)
#sns.lineplot(x=x,y=ysmooth,label='smooth')
sns.lineplot(x=x,y=yhat,label='$\hat f(x)$',ax=ax)
#ax = plt.gca().twinx()
ax = sns.lineplot(x=x,y=0.1*grads,label=r'$\nabla \hat f(x)$',ax=ax)
ax.lines[2].set_linestyle("--")
plt.legend(ncol=3)
plt.savefig('out/toy.pdf')
plt.show()

#%%
# % Let us begin with a simple toy model showing our approach.
# % Assume we wish to minimize the piece-wise constant function presented in Figure~\ref{fig:toy}
# % \begin{figure}
# %     \centering
# %     \includegraphics{fig/toy.pdf}
# %     \caption{Toy model}
# %     \label{fig:toy}
# % \end{figure}
# % Since the gradient of $f$ 0 almost everywhere, gradient methods cannot be applied.
# % However, we can approximate the staircase function $f(x)$ a neural network $\hat f(x)$.
# % The NN is differentiable, so we can surrogate the objective gradient by the gradient of the approximation $\nabla \hat f(x)$ in gradient descent.
# % Notice that sign of the surrogate gradient shows the correct monotonicity of the function.


