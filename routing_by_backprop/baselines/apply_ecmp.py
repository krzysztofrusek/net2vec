import argparse
import pickle

import networkx as nx
import numpy as np


# Sample script that takes as input a nx graph and a TM and applies ECMP

class ECMP:
    ecmp_routing_matrix = None
    next_hop_dict = None
    G = None
    traffic_matrix = None

    def __init__(self, graph_file, TM):
        self.traffic_matrix = TM
        self.G = graph_file
        self.apply_ecmp()

    # Applied ecmp to a given TM and a nx Graph
    def apply_ecmp(self):
        visited_pairs = set()
        # For all pairs of nodes, create the next hop set()
        self.next_hop_dict = {i : {j : set() for j in range(self.G.number_of_nodes()) if j != i} for i in range(self.G.number_of_nodes())}
        # Iterate over all pairs of nodes
        for src in range(self.G.number_of_nodes()):
            for dst in range(self.G.number_of_nodes()):
                if src == dst: continue
                if (src,dst) not in visited_pairs:
                    routings = set([item for sublist in [[(routing[i],routing[i+1]) for i in range(len(routing)-1)] for routing in nx.all_shortest_paths(self.G, src, dst, 'weight')] for item in sublist])
                    for (new_src,next_hop) in routings:
                        self.next_hop_dict[new_src][dst].add(next_hop)
                        visited_pairs.add((new_src,dst))
                
                traffic = self.traffic_matrix[src][dst]
                self.successive_equal_cost_multipaths(src, dst, traffic)
        
        self._normalize_traffic()

    def successive_equal_cost_multipaths(self, src, dst, traffic):
        new_srcs = self.next_hop_dict[src][dst]
        traffic /= len(new_srcs)
        for new_src in new_srcs:
            self.G[src][new_src]['traffic'] += traffic
            if new_src != dst:
                self.successive_equal_cost_multipaths(new_src, dst, traffic)

    def _normalize_traffic(self):
        for (i,j) in self.G.edges():
            self.G[i][j]['traffic'] /= self.G[i][j]['capacity']


def max_load_ecmp(nx_graph:nx.Graph,tm:np.array)->float:
    scaling_factor = 1000000
    for edge in nx_graph.edges():
        nx_graph.edges[edge]['capacity'] = 1*scaling_factor
        nx_graph.edges[edge]['weight'] = 1
        nx_graph.edges[edge]['traffic'] = 0

    ECMP(nx_graph, (scaling_factor*tm).astype(int))

    max_Uti = 0
    for (i,j) in nx_graph.edges():
        if nx_graph[i][j]['traffic']>max_Uti:
            max_Uti = nx_graph[i][j]['traffic']

    return max_Uti



if __name__ == '__main__':
    # python apply_ecmp.py -d ../gdown/out/opt_sp_janos/opt.pickle -t 0
    parser = argparse.ArgumentParser(description='Input')
    parser.add_argument('-d', help='pickle dir', required=True)
    parser.add_argument('-t', help='indicate tm id',  type=int, required=True)
    args = parser.parse_args()

    # How much we scaled the original TM before we executed DEFO.
    # Remember that DEFO works with INTs and we need to scale all decimal values to INTs.
    scaling_factor = 1000000

    infile = open(args.d,'rb')
    new_dict = pickle.load(infile)
    infile.close()

    # for key, value in new_dict.items():
    #     # do something with keys and values
    #     print(key)
    # print(new_dict["finals"])

    nx_graph = new_dict["graph"]

    # Small preprocessing
    for edge in nx_graph.edges():
        nx_graph.edges[edge]['capacity'] = 1*scaling_factor
        nx_graph.edges[edge]['weight'] = 1
        nx_graph.edges[edge]['traffic'] = 0

    # Scale the TM to match DEFO results
    tm = new_dict["demands"][args.t]*scaling_factor
    # We convert the scaled TM to integers
    ECMP(nx_graph, tm.astype(int))

    max_Uti = 0
    for (i,j) in nx_graph.edges():
        if nx_graph[i][j]['traffic']>max_Uti:
            max_Uti = nx_graph[i][j]['traffic']
    
    print(max_Uti)

