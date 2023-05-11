from src.htstabilizer.connectivity_support import *
import unittest


class TestConnectivitySupport(unittest.TestCase):

    def test_assert_connectivity_is_supported(self):
        for num_qubits, connectivity in get_available_connectivities():
            assert_connectivity_is_supported(num_qubits, connectivity)  # type: ignore

        self.assertRaises(AssertionError, assert_connectivity_is_supported, 1, "linear")
        self.assertRaises(AssertionError, assert_connectivity_is_supported, 2, "linear")
        self.assertRaises(AssertionError, assert_connectivity_is_supported, 7, "linear")

    def test_get_connectivity_graph(self):
        for num_qubits, connectivity in get_available_connectivities():
            with self.subTest(num_qubits=num_qubits, connectivity=connectivity):
                get_connectivity_graph(num_qubits, connectivity)  # type: ignore

        # get_connectivity_graph(5, "T").draw(show=True)
