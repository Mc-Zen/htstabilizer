from src.htcircuits.binary import Binary


import unittest
import numpy as np


class TestBinary(unittest.TestCase):

    def test_constructor(self):
        b = Binary(3)
        b = Binary(0)
        print(b)
        print(b^b, type(b))