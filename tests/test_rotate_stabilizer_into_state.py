from src.htstabilizer.rotate_stabilizer_into_state import *
from src.htstabilizer.stabilizer import Stabilizer

import unittest


class TestDoPrepareSameState(unittest.TestCase):

    def test_same_state(self):
        qc = QuantumCircuit(3)
        qc.s([1, 2])
        qc.h(1)
        qc.cx(1, 2)
        assert_same_state(qc, qc)

    def test_not_same_state(self):
        qc = QuantumCircuit(3)
        qc.s([1, 2])
        qc.h(1)
        qc.cx(1, 2)
        qc1 = qc.copy()
        qc.h(0)
        self.assertRaises(AssertionError, assert_same_state, qc, qc1)


class TestRotateStabilizerIntoState(unittest.TestCase):

    def get_different_bell_circuits(self):
        qc = QuantumCircuit(2)
        qc.h(range(2))
        qc.cz(0, 1)

        qc2 = qc.copy()
        qc2.s(1)
        qc2.h(0)
        qc2.s(0)
        qc2.h(0)
        return qc, qc2

    def test_basic(self):
        qc1, qc2 = self.get_different_bell_circuits()
        self.assertTrue(Stabilizer(qc2).is_equivalent(Stabilizer(qc1)))
        self.assertFalse(do_prepare_same_state(qc1, qc2))

        qc3 = rotate_stabilizer_into_state(qc1, qc2)
        assert_same_state(qc2, qc3)

    def test_basic_inplace(self):
        qc1, qc2 = self.get_different_bell_circuits()
        self.assertTrue(Stabilizer(qc2).is_equivalent(Stabilizer(qc1)))
        self.assertFalse(do_prepare_same_state(qc1, qc2))

        qc3 = rotate_stabilizer_into_state(qc1, qc2, inplace=True)
        self.assertTrue(qc3 is qc1)

        assert_same_state(qc1, qc2)
