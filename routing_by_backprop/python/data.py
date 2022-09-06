import hashlib
import os
import pickle
import random
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jraph
import networkx as nx
import numpy as np
from scipy.stats import beta

import util


class GraphProvider:
    def get(self):
        G = self._get()
        G = nx.convert_node_labels_to_integers(G)
        return G

    def get_config(self):
        pass


class BarabasiAlbert(GraphProvider):
    def __init__(self, n):
        self.n = n
        self.nmin = 10
        self.m = 2

    def _get(self):
        return nx.barabasi_albert_graph(np.random.randint(self.nmin, self.n), self.m)

    def get_config(self):
        return (self.n, self.m, self.nmin)


class ErdosReni(GraphProvider):
    def __init__(self, n):
        self.n = n
        self.p = 2.0 / n

    def _get(self):
        G = nx.fast_gnp_random_graph(self.n, self.p, directed=False)
        largest_cc = max(nx.connected_components(G), key=len)
        Gm = G.subgraph(largest_cc)
        return Gm

    def get_config(self):
        return (self.n,)


class SNDLib(GraphProvider):
    def __init__(self, flist:list):
        self.sndlib_networks = {os.path.split(f)[1][0:-8]: nx.read_graphml(f) for f in flist}
        self.names = list(self.sndlib_networks.keys())

    def _get(self):
        name = np.random.choice(self.names)
        Gm = nx.Graph(self.sndlib_networks[name])
        Gm.graph['sndlib']=name
        return Gm


def jackson_delay(G: nx.Graph) -> float:
    def filter_edge(n1, n2):
        return G[n1][n2]['c'] != 0

    V = nx.subgraph_view(G, filter_edge=filter_edge)

    R = make_routing(V, 'weight')
    demands = []
    for i in V.nodes:
        for j in V.nodes:
            if i != j:
                demands.append(G[i][j]['demand'])
    demands = np.array(demands)

    lambdas = demands @ R
    C = np.array([e['c'] for _, _, e in V.edges.data()])
    rho = lambdas / C
    if np.any(rho > 1):
        raise Exception(f'Unstable system for w=1, {np.sum(rho > 1)} out of {len(V.edges)} overloaded')

    # TODO M/M/1/b fixed point

    L = rho / (1 - rho)
    W_av = L.sum() / demands.sum()
    return W_av


def make_routing(G: nx.DiGraph, weight: str) -> np.ndarray:
    n = len(G)
    edges = list(G.edges)
    sparse_paths = []
    all_pairs = dict(nx.all_pairs_dijkstra_path(G, weight=weight))
    for i in G.nodes:
        for j in G.nodes:
            if i != j:
                list_p = [edges.index(tup) for tup in util.pairwise(all_pairs[i][j])]
                p = np.zeros(len(edges))
                p[list_p] = 1
                sparse_paths.append(p)
    R = np.array(sparse_paths)
    return R


def make_demands(TM: np.ndarray) -> np.ndarray:
    n = TM.shape[0]
    demands = []
    for i in range(n):
        for j in range(n):
            if i != j:
                demands.append(TM[i, j])
    return np.array(demands)


def calc_link_capacities(lambdas: np.ndarray) -> np.ndarray:
    allowed_C = np.array([0.01, 0.1, 0.4, 1.])  # 100Gb/s
    bins = np.concatenate([
        allowed_C,
        [np.inf]
    ])
    c_idx = np.digitize(lambdas, bins=bins, right=False)
    C = allowed_C[np.where(c_idx > 3, 3, c_idx)]
    return C


def random_beta_w(G: nx.Graph) -> np.ndarray:
    # random weights 0.5, 1.5 with mode at 1
    w_dist = beta(a=10, b=10, scale=2)
    w = w_dist.rvs(len(G.edges))
    return w


def uniform_tm(G: nx.Graph) -> np.ndarray:
    n = len(G)
    T = 1.
    TM = T * np.random.uniform(0.1, 1., size=(n, n)) / (n - 1)
    return TM

def gravity_tm(G: nx.Graph) -> np.ndarray:
    n = len(G)
    Tin = np.random.exponential(size=(n, 1))
    Tout = np.random.exponential(size=(n, 1))

    pin = Tin / Tin.sum()
    pout = Tout / Tout.sum()
    Pgr = pin * pout.T
    # scale models fluctuations during busy h
    loc = 0.1*n ** 0.1
    scale = 0.1 * loc
    T = np.random.normal(loc=loc, scale=scale)
    TM = T * Pgr
    return TM

def make_sample(provider: GraphProvider,
                w_fn=random_beta_w,
                tm_fn=uniform_tm):
    G = provider.get()
    G = nx.DiGraph(G)

    TM = tm_fn(G)

    C = np.ones(len(G.edges))
    nx.set_edge_attributes(G, {e: c for e, c in zip(G.edges, C)}, name='c')

    w = {k: v for k, v in zip(G.edges, w_fn(G))}
    nx.set_edge_attributes(G, w, name='weight')

    paths = []
    edges = list(G.edges)
    all_pairs = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))

    for i in G.nodes:
        for j in G.nodes:
            if i != j:
                paths.append(list(map(edges.index, util.pairwise(all_pairs[i][j]))))

    net_tup = jaxon.NetTuple(
        A=make_demands(TM),
        c=C,
        R=paths
    )

    p_net_tup = jaxon.paded(net_tup,
                            _nearest_bigger_power_of_two(len(edges)),
                            _nearest_bigger_power_of_two(len(paths))
                            )


    rhos = jaxon.rho(p_net_tup)
    unpaded_rho = rhos[jnp.where(jnp.isfinite(p_net_tup.c))]

    # Make full graph and end encode demands
    for src in G.nodes:
        for dst in G.nodes:
            if src != dst:
                if not (src, dst) in G.edges:
                    G.add_edge(src, dst, c=0, weight=np.inf, y=0)
                else:
                    nx.set_edge_attributes(G, {(src, dst): unpaded_rho[edges.index((src, dst))]}, name='y')
                nx.set_edge_attributes(G, {(src, dst): TM[src, dst]}, name='demand')

    return G


class Label(NamedTuple):
    label: jnp.ndarray
    predicted: jnp.ndarray


class NodeData(NamedTuple):
    weight: jnp.ndarray
    c: jnp.ndarray
    demand: jnp.ndarray
    physical: jnp.ndarray
    y: jnp.ndarray


class Scaler(NamedTuple):
    mean: jnp.ndarray
    std: jnp.ndarray


def graph2graphs_tuple(G: nx.DiGraph):
    # Links are nodes
    L = nx.line_graph(G)

    assert G.edges == L.nodes

    # Link features become node features
    fs = []
    for s, r, f in G.edges.data():
        fs.append(NodeData(
            weight=f['weight'],
            c=f['c'],
            demand=f['demand'],
            physical=jnp.ones_like(f['weight']),
            y=f['y']
        ))
    fs = NodeData(*jnp.split(jnp.array(fs), len(fs[0]), axis=1))
    fin_idx = jnp.where(jnp.isfinite(fs.weight),jnp.ones_like(fs.weight), jnp.zeros_like(fs.weight))
    fs = fs._replace(
        weight=jnp.where(fin_idx>0,fs.weight, jnp.ones_like(fs.weight)),
        physical=fin_idx
    )
    # Remapping indices
    e = list(G.edges)
    senders, receivers = jnp.split(
        jnp.asarray([(e.index(s), e.index(r)) for s, r in L.edges]),
        2, axis=1)

    return jraph.GraphsTuple(
        n_node=jnp.asarray([len(L)]),
        n_edge=np.asarray([len(L.edges)]),
        nodes=fs,
        edges=None,
        senders=senders.flatten(),
        receivers=receivers.flatten(),
        globals=None
    )


def _nearest_bigger_power_of_two(x: int) -> int:
    """Computes the nearest power of two greater than x for padding.
  Idea taken from:
   https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train_flax.py
   """
    y = 2
    while y < x:
        y *= 2
    return y


@dataclass
class Dataset:
    cache_filename: str
    provider: GraphProvider
    len: int
    batch_size: int = 8
    data = []

    @property
    def config_hash(self):
        config = (self.len, self.cache_filename) + self.provider.get_config()
        md5 = hashlib.md5(pickle.dumps(config))
        return md5.hexdigest()

    @property
    def stats_filename(self):
        return self.cache_filename + '_stats_' + self.config_hash

    def __iter__(self):
        if not self.data:
            fname = self.cache_filename + self.config_hash
            if os.path.exists(fname):
                with open(fname, 'rb') as f:
                    self.data = pickle.load(f)
            else:
                self.data = [graph2graphs_tuple(make_sample(self.provider)) for _ in range(self.len)]
                with open(fname, 'wb') as f:
                    pickle.dump(self.data, f)
        return self

    def __next__(self):
        samples = random.choices(self.data, k=self.batch_size)
        batch = jraph.batch(samples)

        # Add 1 since we need at least one padding node for pad_with_graphs.
        pad_nodes_to = _nearest_bigger_power_of_two(jnp.sum(batch.n_node)) + 1
        pad_edges_to = _nearest_bigger_power_of_two(jnp.sum(batch.n_edge))
        # Add 1 since we need at least one padding graph for pad_with_graphs.
        # We do not pad to nearest power of two because the batch size is fixed.
        pad_graphs_to = batch.n_node.shape[0] + 1

        padded_batch = jraph.pad_with_graphs(batch,
                                             n_node=pad_nodes_to,
                                             n_edge=pad_edges_to,
                                             n_graph=pad_graphs_to)

        return padded_batch

    def get_normalizers(self):
        if os.path.exists(self.stats_filename):
            with open(self.stats_filename, 'rb') as f:
                stats = pickle.load(f)
            stats = stats._replace(globals=Label(None, None))
        else:
            iter(self)
            wb = util.batched_welford()

            mask = jax.tree_map(lambda x: None, self.data[0])
            mask = mask._replace(nodes=NodeData(False, False, False, False, False))

            state = jax.tree_multimap(lambda x, m: wb.init(jnp.zeros_like(x[0])) if m else None, self.data[0], mask)

            for d in self.data:
                state = jax.tree_multimap(lambda s, x, m: wb.update(s, x) if m else None, state, d, mask,
                                          is_leaf=lambda x: type(x) in [util.WelfordState])

            stats = jax.tree_map(wb.stats, state, is_leaf=lambda x: type(x) in [util.WelfordState])
            with open(self.stats_filename, 'wb') as f:
                pickle.dump(stats, f)

        def normalize(batch):
            return jax.tree_multimap(lambda x, y: (x - y.mean) / y.std if y else x,
                                     batch,
                                     stats)

        def denormalize(batch):
            return jax.tree_multimap(lambda x, y: x * y.std + y.mean if y else x,
                                     batch,
                                     stats)

        return normalize, denormalize
