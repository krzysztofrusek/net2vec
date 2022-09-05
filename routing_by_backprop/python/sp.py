import datetime
import functools
import os
import pickle
from dataclasses import dataclass
from typing import Iterable, Tuple, Any

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import networkx as nx
import numpy as np
import optax
from absl import flags, app, logging

import data
import util

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_steps", 100000, "...")
flags.DEFINE_integer("checkpoint_steps", 2000, "...")
flags.DEFINE_integer("units", 16, "...")
flags.DEFINE_string('tag', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), "Name of the run")
flags.DEFINE_string('directory', "log", "Model directory for logs an checkpoints")
flags.DEFINE_float('learning_rate', 0.001, "Learning rate")
flags.DEFINE_integer('eval_steps', 4, "...")  # epoch
flags.DEFINE_integer('train_graphs', 30, "...")
flags.DEFINE_integer('test_graphs', 3, "...")
flags.DEFINE_string('hot_start_from', None, 'starting checkpoint')
flags.DEFINE_integer('batch_size', 8, "...")


@dataclass
class InfiniteDataset:
    provider: data.GraphProvider
    batch_size:int=8

    def random_weights(self,G:nx.Graph):
        return data.random_beta_w(G)

    def make_graph(self):
        G = self.provider.get()
        G = nx.DiGraph(G)

        w = {k: v for k, v in zip(G.edges, self.random_weights(G))}
        nx.set_edge_attributes(G, w, name='weight')

        all_pairs = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))
        G.graph['all_pairs']=all_pairs
        return G

    def make_sample(self, graph=False):
        G = self.make_graph()
        nnodes = len(G)
        routing = G.graph['all_pairs']
        ledges = list(G.edges)
        relation_data = np.array(list(G.edges.data('weight')))

        src, dst = np.random.choice(nnodes, 2, replace=False)

        nodes = np.zeros((nnodes, 2))
        nodes[src, 0] = 1
        nodes[dst, 1] = 1
        nodes = jnp.array(nodes)


        senders, receivers = jnp.split(relation_data[:, :2].astype(np.int32), 2, 1)
        edges = jnp.array(relation_data[:, 2:])

        x = jraph.GraphsTuple(
            n_node=jnp.asarray([nodes.shape[0]]),
            n_edge=jnp.asarray([edges.shape[0]]),
            nodes=nodes,
            edges=edges,
            senders=senders.flatten(),
            receivers=receivers.flatten(),
            globals=None
        )
        y = np.zeros((relation_data.shape[0],1))

        path = [ledges.index(e) for e in util.pairwise(routing[src][dst])]
        y[path,0] = 1

        if graph:
            return x,y,G
        else:
            return x,y

    def make_padded_batch(self):
        xy = [self.make_sample() for _ in range(self.batch_size)]
        bx = jraph.batch([x for x,y in xy])
        by=jnp.concatenate([y for x,y in xy])

        # Add 1 since we need at least one padding node for pad_with_graphs.
        pad_nodes_to = data._nearest_bigger_power_of_two(jnp.sum(bx.n_node)) + 1
        pad_edges_to = data._nearest_bigger_power_of_two(jnp.sum(bx.n_edge))
        # Add 1 since we need at least one padding graph for pad_with_graphs.
        # We do not pad to nearest power of two because the batch size is fixed.
        pad_graphs_to = bx.n_node.shape[0] + 1

        pbx = jraph.pad_with_graphs(bx,
                                             n_node=pad_nodes_to,
                                             n_edge=pad_edges_to,
                                             n_graph=pad_graphs_to)
        pby = jnp.pad(by,((0,pad_edges_to-by.shape[0]),(0,0)))
        return pbx, pby


    def __call__(self, *args, **kwargs):
        yield self.make_padded_batch()

NUM_LAYERS = 2  # Hard-code number of layers in the edge/node/global models.
LATENT_SIZE = 128  # Hard-code latent layer sizes for demos.


class MLPModule(hk.Module):
  def __call__(self, x, is_training=False)->jnp.ndarray:
    x=hk.nets.MLP([LATENT_SIZE] * NUM_LAYERS,activate_final=True)(x)
    x = hk.LayerNorm(axis=1, create_offset=True, create_scale=True)(x)
    return x


def network_definition(graph: jraph.GraphsTuple, is_training: bool = True) -> jraph.GraphsTuple:
    """`InteractionNetwork` with an GRU in the node update.
    This is a simple message passing based on https://github.com/deepmind/jraph/blob/master/jraph/examples/lstm.py
    """

    latent = jraph.GraphMapFeatures(MLPModule(), MLPModule())(graph)
    latent0 = latent
    outputs = []

    core = jraph.GraphNetwork(
        update_edge_fn=jraph.concatenated_args(MLPModule()),
        update_node_fn=jraph.concatenated_args(MLPModule())
    )
    decoder = jraph.GraphMapFeatures(MLPModule(), MLPModule())

    output_transform = jraph.GraphMapFeatures(
        embed_edge_fn=hk.Linear(1)
    )

    num_message_passing_steps = 8
    for _ in range(num_message_passing_steps):
        core_input = latent._replace(
            nodes=jnp.concatenate((latent.nodes, latent0.nodes), axis=1),
            edges=jnp.concatenate((latent.edges, latent0.edges), axis=1)
        )
        latent = core(core_input)
        decoded = decoder(latent)
        outputs.append(output_transform(decoded))

    return outputs


def main(argv):
    ds = InfiniteDataset(data.BarabasiAlbert(20))
    val_ds = InfiniteDataset(data.BarabasiAlbert(20))

    x,y = next(ds())

    network = hk.without_apply_rng(hk.transform_with_state(network_definition))

    params, state = network.init(jax.random.PRNGKey(42), x)

    opt = optax.adam(FLAGS.learning_rate)

    @functools.partial(jax.jit, static_argnums=4) #4th argument is static
    def loss(params: hk.Params, state: hk.State, x:jraph.GraphsTuple, y:jnp.ndarray, is_training=True) -> Tuple[Any, Any]:
        hats, state = network.apply(params, state, x, is_training=is_training)
        mask = jraph.get_edge_padding_mask(x)[..., jnp.newaxis]

        losses = [jnp.mean(optax.sigmoid_binary_cross_entropy(hat.edges, y), where=mask) for hat in hats]
        final_loss = sum(losses)/len(losses) if is_training else losses[-1]
        return final_loss, state

    @jax.jit
    def update(params, state, opt_state, x,y):
        (l, s), grads = jax.value_and_grad(loss, has_aux=True)(params, state, x,y, True)
        updates, new_opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return l, s, new_params, new_opt_state

    opt_state = opt.init(params)

    logdir = os.path.join(FLAGS.directory, FLAGS.tag)
    try:
        os.makedirs(logdir)
    except Exception:
        pass
    FLAGS.append_flags_into_file(os.path.join(logdir, 'flagfile.txt'))

    tb = util.TensorBoard(logdir=logdir)

    def evaluate(params: hk.Params, net_state: hk.State, val_ds: Iterable, steps: int):
        b = next(val_ds())
        l, _ = loss(params, net_state, *b, False)
        w = util.welford()
        state = w.init(l)
        for _ in range(steps):
            l,_ = loss(params,net_state, *next(val_ds()),False)
            state = w.update(state, l)
        return w.stats(state)

    num_val = FLAGS.test_graphs

    for step in range(FLAGS.num_steps):
        batch = next(ds())
        if step % FLAGS.eval_steps == 0:
            # Periodically evaluate the model on train & test sets.
            l_stats = evaluate(params, state, val_ds, num_val)
            logging.info(f"eval mean loss {l_stats.mean} +- (std) {l_stats.std}")
            tb.scalar("val", l_stats.mean, step)
            tb.scalar("val_std", l_stats.std, step)
        # Do SGD on a batch of training examples.
        l, state, params, opt_state = update(params, state, opt_state, *batch)
        logging.info(f"train {step} {l}")
        tb.scalar("train", l, step)

        if step % FLAGS.checkpoint_steps == 0:
            with open(os.path.join(logdir, f'snapshot-{step}.pickle'), 'wb') as f:
                pickle.dump([params, state], f)


    return 0


if __name__ == '__main__':
    app.run(main)
