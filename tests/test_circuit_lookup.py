import unittest

from src.htcircuits.circuit_lookup import *

class TestCircuitLookup(unittest.TestCase):

    def test_circuit_lookup(self):
        for i in range(1000):
            circuit_lookup(3, "linear")
            circuit_lookup(3, "linear")
            circuit_lookup(3, "linear")
            circuit_lookup(3, "linear")
            circuit_lookup(3, "linear")
            circuit_lookup(3, "linear")
        
    def test_c(self):
        for i in range(93):
            qc = circuit_lookup(5, "linear", i)
            # print(qc.draw("text"))