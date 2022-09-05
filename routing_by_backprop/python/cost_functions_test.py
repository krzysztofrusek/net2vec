import unittest
import numpy as np
import cost_functions as cf

class TestCase(unittest.TestCase):
    def test_sumsoftmax(self):
        x = np.asarray([0.2,0.1,0.9])
        m = cf.sumsoftmax(0.1)(x)
        self.assertTrue(np.isclose(m,max(x),rtol=0.05))
    def test_queue(self):
        x = np.asarray([0.2, 0.1, 0.9])
        y=cf.average_queue(4)(x)
        self.assertTrue(np.isclose(y[0],0.248399))


if __name__ == '__main__':
    unittest.main()
