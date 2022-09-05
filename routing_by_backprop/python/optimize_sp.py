import pickle
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import networkx as nx
import numpy as np
import pandas as pd
from absl import flags, app, logging

import cost_functions as cf
import data
import sp
import util
from apply_ecmp import ECMP

# from data import *
# sys.path.append('/net/software/local/python/3.9/lib/python3.9/site-packages')

flags.DEFINE_integer("num_opt", 3, "...")
flags.DEFINE_integer("net_size", 20, "Number of nodes")
flags.DEFINE_string("opt_checkpoint", 'log/labsim/sp1/snapshot-8000.pickle', "params")
flags.DEFINE_string("report", 'opt_janos.csv', "outpu file")
flags.DEFINE_string("sndlib", 'topo/sndlib-networks-xml/janos-us.graphml', "SNDLib topology file")
flags.DEFINE_float('tmscale',0.6, "scaling factor for demands")
flags.DEFINE_integer("num_sgd", 4, "...")
flags.DEFINE_bool("pmap", True, "use pmap, otherwise jit + vmap")
flags.DEFINE_integer("lsearch", 2, "number of line search steps")

FLAGS = flags.FLAGS





def load(start_from):
    with open(start_from, 'rb') as f:
        params, state = pickle.load(f)
    logging.info(f'loaded {start_from}')
    return params, state


#Alternative: cost_fn = cf.average_queue(4)
cost_fn = cf.sumsoftmax(0.1)

def true_cost(H: nx.DiGraph, tm: np.array, src: np.array, dst: np.array, x: jraph.GraphsTuple, w: np.array,
              return_delays=False):
    edge_attributes = {(int(s), int(r)): float(aw) for s, r, aw in zip(x.senders, x.receivers, w)}
    nx.set_edge_attributes(H, edge_attributes, 'weight')
    all_pairs = dict(nx.all_pairs_dijkstra_path(H, weight='weight'))

    hard_routing = []
    ledges = list(H.edges)
    for sd in zip(src, dst):
        s,d = jax.tree_map(int,sd)
        row = np.zeros(w.shape[0])
        path = [ledges.index(e) for e in util.pairwise(all_pairs[s][d])]
        row[path] = 1
        hard_routing.append(row)

    hard_routing = np.stack(hard_routing, axis=0)
    flat_tm = tm[src, dst]
    link_traffic = flat_tm @ hard_routing
    c = 1.
    th = float(np.max(link_traffic))

    if return_delays:
        return th, link_traffic
    return th

def true_cost_with_ecmp(H: nx.DiGraph, tm: np.array, x: jraph.GraphsTuple, w: np.array):
    edge_attributes = {(s, r): float(aw) for s, r, aw in zip(x.senders, x.receivers, w)}
    nx.set_edge_attributes(H, edge_attributes, 'new_weight')

    n = len(H)
    nx.set_edge_attributes(H, 0, 'traffic')
    nx.set_edge_attributes(H,int(1e6),'capacity')
    ecmp = ECMP(H, (1e6*tm).astype(np.int32))
    ecmp.apply_ecmp()
    max_util = max([u for _, _, u in H.edges.data('traffic')])
    return float(max_util)


def main(_):
    params, state = load(FLAGS.opt_checkpoint)

    ds = sp.InfiniteDataset(data.SNDLib([FLAGS.sndlib]))
    x, y, G = ds.make_sample(graph=True)
    network = hk.without_apply_rng(hk.transform_with_state(sp.network_definition))
    hats, _ = network.apply(params, state, x, is_training=False)

    n = len(G)

    def apply_for_pair(w: jnp.array, src: jnp.array, dst: jnp.array, params: hk.Params, state: hk.State,
                       graph_template: jraph.GraphsTuple) -> jnp.array:
        nodes = jnp.zeros_like(graph_template.nodes)
        nodes = nodes.at[src, 0].set(1)
        nodes = nodes.at[dst, 1].set(1)
        graph = graph_template._replace(nodes=nodes)
        graph = graph._replace(edges=w)
        hats, _ = network.apply(params, state, graph, is_training=False)
        return jax.nn.sigmoid(hats[-1].edges.flatten())

    apply_for_graph = jax.vmap(apply_for_pair, in_axes=(None, 0, 0, None, None, None))


    @jax.jit
    def cost(w: jnp.array, src: jnp.array, dst: jnp.array, params: hk.Params, state: hk.State,
             graph_template: jraph.GraphsTuple, tm: jnp.array) -> jnp.array:
        p = apply_for_graph(w, src, dst, params, state, graph_template)
        flat_tm = tm[src, dst]
        link_traffic = flat_tm @ p
        c = 1.  # uniform network
        return cost_fn(link_traffic)


    if FLAGS.pmap:
        lrs = jnp.asarray([1e-2,1e-3])
    else:
        if FLAGS.lsearch >2:
            lrs = jnp.logspace(-8., 0.0, FLAGS.lsearch)
        elif FLAGS.lsearch ==2:
            lrs = jnp.asarray([1e-2, 1e-3])
        else:
            lrs = jnp.asarray([2e-2])


    def update(w: jnp.array, src: jnp.array, dst: jnp.array, params: hk.Params, state: hk.State,
             graph_template: jraph.GraphsTuple, tm: jnp.array,lr: chex.Array) -> jnp.array:
        grads = jax.grad(cost)(w, src, dst, params, state, graph_template, tm)
        proposal = w - lr*grads
        proposal = jnp.where(proposal < 0, 3*lr, proposal)
        value = cost(proposal,src, dst, params, state, graph_template, tm)
        return value, proposal

    def fast_update(w: jnp.array, src: jnp.array, dst: jnp.array, params: hk.Params, state: hk.State,
               graph_template: jraph.GraphsTuple, tm: jnp.array, lr: chex.Array) -> jnp.array:
        grads = jax.grad(cost)(w, src, dst, params, state, graph_template, tm)
        proposal = w - lr * grads
        proposal = jnp.where(proposal < 0, 3 * lr, proposal)
        return  proposal,proposal

    @jax.jit
    def select(values: chex.Array, proposals: chex.Array) -> chex.Array:
        min_idx = jnp.argmin(values)
        return proposals[min_idx, ...],min_idx

    @jax.jit
    def fast_select(values: chex.Array, proposals: chex.Array) -> chex.Array:
        return proposals,0

    srcu, dstu = np.triu_indices(n, k=1)
    srcl, dstl = np.tril_indices(n, k=-1)

    src = jnp.concatenate([srcu, srcl], axis=0)
    dst = jnp.concatenate([dstu, dstl], axis=0)

    if FLAGS.pmap:
        pupdate = jax.pmap(update, in_axes=(None,) + 5 * (0,) + (None, 0))
        devs = jax.local_devices()
        slrs = jax.device_put_sharded(list(lrs),devs)

        ssrc = jax.device_put_replicated(src,devs)
        sdst = jax.device_put_replicated(dst, devs)
        sparams = jax.device_put_replicated(params, devs)
        sstate = jax.device_put_replicated(state, devs)
        sx = jax.device_put_replicated(x, devs)
    else:
        pupdate = jax.jit(jax.vmap(update, in_axes=7*(None,) + (0,)))
        if len(lrs)<2:
            pupdate = jax.jit(fast_update)
            select = fast_select

        slrs = lrs
        ssrc = src
        sdst = dst
        sparams = params
        sstate = state
        sx = x

    results:list = []
    tms:list = []
    finals:list=[]
    np.random.seed(127445)#random.org

    for problem in range(FLAGS.num_opt):
        logging.info(f'Problem {problem}')
        w = jnp.ones_like(x.edges)
        tm = FLAGS.tmscale * np.random.uniform(0.0, 1.0, size=(n, n)) / (n - 1)

        tms.append(tm)
        tm = jnp.array(tm)

        H = G.copy()
        initialcost, new_w = true_cost(H, tm, src, dst, x, w, return_delays=True)
        #ecmp = true_cost_with_ecmp(H.copy(), tm, x, w)
        # baselinecost = true_cost(H, tm, src, dst, x, 1./new_w.T, return_delays=False)
        result = dict(
            initialcost=initialcost,
            #initialecmp=ecmp
            # baselinecost = baselinecost
        )

        for i in range(FLAGS.num_sgd):
            logging.info('begin')
            # update(w, ssrc[0,...], sdst[0,...], sparams[0,...], sstate[0,...], sx[0,...], tm,slrs[0,...])
            # update(w, src, dst, params, state, x, tm, lrs[0])
            with util.timer() as pt:
                v, p = pupdate(w, ssrc, sdst, sparams, sstate, sx, tm,slrs)
                w,mini = select(v, p)
                w=w.block_until_ready()

            logging.info('end')
            logging.info(f'mini {mini}')
            result[f'finalcost_{i}'] = true_cost(H, tm, src, dst, x, w)
            result[f'time_ns_{i}'] = pt.diff()
            result[f'minlr_{i}'] = int(mini)
            logging.info(result[f'finalcost_{i}'])

        # result['finalcost'] = true_cost(H, tm, src, dst, x, w)
        # if problem < 100:
        #     result['finalcostecmp'] = true_cost_with_ecmp(H, tm, x, w)
        # else:
        #     result['finalcostecmp']=np.nan
        #result['finalcostecmp'] = true_cost_with_ecmp(H, tm, x, w)

        logging.info(result)
        results.append(result)
        finals.append(H)

    opt_stats = pd.DataFrame(results)
    opt_stats.to_csv(FLAGS.report)

    with open(FLAGS.report.replace('csv','pickle'),'wb') as f:
        pickle.dump(dict(graph=G, demands=tms, finals=finals),f, protocol=pickle.HIGHEST_PROTOCOL)

    return 0


if __name__ == '__main__':
    app.run(main)
