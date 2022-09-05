import glob
import unittest

import networkx as nx


class TestCase(unittest.TestCase):
    def test_attr(self):
        for gml in glob.glob('sndlib-networks-xml/*.graphml'):
            G = nx.read_graphml(gml)

            self.assertTrue(len(nx.get_edge_attributes(G,'cost'))>0)
            self.assertTrue(len(nx.get_edge_attributes(G,'capacity'))>0)


if __name__ == '__main__':
    unittest.main()
