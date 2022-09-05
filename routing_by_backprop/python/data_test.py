import unittest
import networkx as nx
import numpy as np
import data

class TestProvider(data.GraphProvider):
    def get(self):
        return nx.DiGraph(nx.Graph([(0, 1), (1, 2)]))

def make_demand_fn(t=0.5):
    def demand(G):
        n=len(G)
        return t*np.ones((n,n))
    return demand

class DataTestCase(unittest.TestCase):
    def test_sample(self):
        provider = TestProvider()
        sample = data.make_sample(provider,tm_fn=make_demand_fn(0.6))
        self.assertEqual(len(sample.edges), 3*(3-1))
        self.assertEqual(sample.edges[(0,2)]['c'], 0)
        self.assertEqual(sample.edges[(2, 0)]['c'], 0)
        self.assertEqual(sample.edges[(1,0)]['y'], 1.2)

        pass


class SNDLibTest(unittest.TestCase):
    def test_load(self):
        provider  = data.SNDLib(['../topo/sndlib-networks-xml/janos-us.graphml'])
        G = provider.get()
        self.assertTrue(G.graph['sndlib']=='janos-us')


if __name__ == '__main__':
    unittest.main()
