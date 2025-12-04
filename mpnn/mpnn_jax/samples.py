import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx
import os
from tqdm import trange
import xml.etree.ElementTree as ET
from pathlib import Path


class GraphProvider:
    def get(self, key):
        """
        Draw a graph instance using the subclass-specific sampler and
        reindex nodes to consecutive integers starting at 0.
        """
        G, new_key = self._get(key)
        G = nx.convert_node_labels_to_integers(G)
        return G, new_key


class BarabasiAlbert(GraphProvider):
    def __init__(self, n):
        self.n = n
        self.nmin = 10
        self.m = 2

    def _get(self, key):
        key, subkey = jax.random.split(key)

        n_nodes = int(
            jax.random.randint(subkey, shape=(), minval=self.nmin, maxval=self.n)
        )

        key, subkey = jax.random.split(key)
        seed = int(jax.random.randint(subkey, (), 0, 2**31 - 1))

        G = nx.barabasi_albert_graph(n_nodes, self.m, seed=seed)

        return G, key


class ErdosReni(GraphProvider):
    def __init__(self, n):
        self.n = n
        self.p = 2.0 / n

    def _get(self, key):
        key, subkey = jax.random.split(key)

        seed = int(jax.random.randint(subkey, (), 0, 2**31 - 1))
        G = nx.fast_gnp_random_graph(self.n, self.p, seed=seed, directed=False)

        largest_cc = max(nx.connected_components(G), key=len)
        Gm = G.subgraph(largest_cc).copy()

        return Gm, key


def read_sndlib_xml(filepath):
    """
    Parse an SNDlib XML file and construct an undirected NetworkX graph.
    """
    G = nx.Graph()
    ns = {'snd': 'http://sndlib.zib.de/network'} 

    tree = ET.parse(filepath)
    root = tree.getroot()

    nodes_element = root.find('snd:networkStructure/snd:nodes', ns)
    if nodes_element is not None:
        for node in nodes_element.findall('snd:node', ns):
            node_id = node.get('id')
            G.add_node(node_id)

    links_element = root.find('snd:networkStructure/snd:links', ns)
    if links_element is not None:
        for link in links_element.findall('snd:link', ns):
            source = link.find('snd:source', ns).text
            target = link.find('snd:target', ns).text
            G.add_edge(source, target)
            
    return G


class SNDLib(GraphProvider):
    """
    Graph provider for SNDlib topologies.
    
    flist may contain XML files and/or directories; directories are
    scanned recursively for .xml files to build the pool of networks.
    """
    def __init__(self, flist):

        all_files = []

        for f in flist:
            if os.path.isdir(f):
                for root, _, files in os.walk(f):
                    for name in files:
                        if name.lower().endswith(".xml"):
                            all_files.append(os.path.join(root, name))
            else:
                all_files.append(f)

        if not all_files:
            raise ValueError(f"No XML files found for SNDLib in: {flist}")

        self.sndlib_networks = {
            os.path.splitext(os.path.basename(path))[0]: read_sndlib_xml(path)
            for path in all_files
        }

        # self.sndlib_networks = {k:v for k,v in self.sndlib_networks.items() if len(v) < 38 and len(v) > 19}

        self.names = list(self.sndlib_networks.keys())

    def _get(self, key):
        key, subkey = jax.random.split(key)
        idx = int(jax.random.randint(subkey, (), 0, len(self.names)))
        name = self.names[idx]
        Gm = nx.Graph(self.sndlib_networks[name])
        if not nx.is_connected(Gm):
            Gm = Gm.subgraph(max(nx.connected_components(Gm), key=len)).copy()
        return Gm, key


def attach_results_to_graph(Gm, mu, L, R, W):
    """
    Store per-node and per-edge attributes, plus global W, into a NetworkX graph.
    """
    nx.set_node_attributes(Gm, dict(enumerate(jnp.ravel(mu))), "mu")
    nx.set_node_attributes(Gm, dict(enumerate(jnp.ravel(L))), "L")

    R_indices = jnp.argwhere(R > 0)
    edge_values = {(int(i), int(j)): float(R[i, j]) for i, j in R_indices}
    nx.set_edge_attributes(Gm, edge_values, "R")

    Gm.graph["W"] = float(W)

    return Gm


def make_sample(provider: GraphProvider, key, rl=0.3, rh=0.7):
    """
    Generate one synthetic queueing-network sample on a given topology:
      1. Sample Λ and normalize to 1.
      2. Build row-normalized routing R from adjacency.
      3. Solve λ for internal arrival rates.
      4. Sample utilizations ρ and set μ = λ / ρ.
      5. Compute Little's-law-based W.
    """
    Graph, key = provider.get(key)
    adjacency_matrix = jnp.array(nx.convert_matrix.to_numpy_array(Graph))
    num_nodes = len(Graph)

    key, subkey = jax.random.split(key)
    L = jax.random.uniform(subkey, shape=(num_nodes, 1))
    L = L / jnp.sum(L)

    node_degree = jnp.sum(adjacency_matrix, axis=1)
    routing_factor = 1.0 / (node_degree + 1.0)
    R = adjacency_matrix * routing_factor[:, None]     

    I = jnp.identity(num_nodes)
    lam = jnp.linalg.solve(I - R.T, L)

    key, subkey = jax.random.split(key)
    utilization = jax.random.uniform(subkey, shape=lam.shape, minval=rl, maxval=rh)

    mu = lam / utilization
    ll = utilization / (1 - utilization)
    W = jnp.sum(ll) / jnp.sum(L)
    #(mu, L, R, W)

    return mu, L, R, W, attach_results_to_graph(Graph, mu, L, R, W), key


def make_dataset(output_dir, provider : GraphProvider, key, train_size, val_size, test_size, rl, rh):
    """
    Generate (train/val/test) splits of synthetic samples and save them to .npz files.
    """
    output_dir = Path(output_dir) / provider.__class__.__name__
    os.makedirs(output_dir, exist_ok=True)

    total = train_size + val_size + test_size

    samples = []
    for _ in trange(total, desc="Generating samples"):
        mu, L, R, W, _, key = make_sample(provider, key, rl, rh)
        samples.append((mu, L, R, W))

    samples = np.array(samples, dtype=object)
    key, subkey = jax.random.split(key)
    idx = jax.random.permutation(subkey, len(samples))
    samples = samples[idx]

    train = samples[:train_size]
    val = samples[train_size : train_size + val_size]
    test = samples[train_size + val_size :]

    def unzip(block):
        mu, L, R, W = zip(*block)
        return list(mu), list(L), list(R), list(W)
    
    def save(path, mu, L, R, W):
        np.savez_compressed(
            path,
            mu=np.array([np.asarray(x, np.float32) for x in mu], dtype=object),
            L=np.array([np.asarray(x, np.float32) for x in L], dtype=object),
            R=np.array([np.asarray(x, np.float32) for x in R], dtype=object),
            W=np.asarray(W, dtype=np.float32),
        )

    datasets_to_process = [
        ("train", train, train_size),
        ("val", val, val_size),
        ("test", test, test_size)
    ]

    for name, data_block, size in datasets_to_process:
        if size > 0:
            mu_data, L_data, R_data, W_data = unzip(data_block)
            save_path = os.path.join(output_dir, f"{name}.npz")
            save(save_path, mu_data, L_data, R_data, W_data)

    print("Dataset generated deterministically.")
    return key


if __name__ == "__main__":
    from config import DatasetConfig, GraphType

    cfg = DatasetConfig()

    key = jax.random.key(cfg.seed)

    provider = None

    match cfg.graph_type:
        case GraphType.BARABASI_ALBERT:
            provider = BarabasiAlbert(cfg.n_nodes)
        case GraphType.ERDOS_RENYI:
            provider = ErdosReni(cfg.n_nodes)
        case GraphType.SNDLIB:
            provider = SNDLib(cfg.snd_path)

    make_dataset(
        cfg.output_dir, provider, key, cfg.train_size, cfg.val_size, cfg.test_size, cfg.rl, cfg.rh
    )

