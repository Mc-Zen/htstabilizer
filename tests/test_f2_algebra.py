from src.htstabilizer.f2_algebra import *


import unittest


class TestF2Algebra(unittest.TestCase):

    def test_rref_and_basis_change(self):

        A = np.array([[1, 1, 0, 1, 0],
                      [1, 0, 1, 0, 0],
                      [1, 0, 0, 0, 1]])

        rref, M, Mi = rref_and_basis_change(A)
        print(mat_mul(Mi, rref))

    def test_rref_and_basis_change_random(self):
        for i in range(1000):
            A = np.round(np.random.random((7, 5))).astype(np.int8)
            rref_A, M, Mi = rref_and_basis_change(A)
            np.testing.assert_array_equal(rref_A, rref(A)[0])
            np.testing.assert_array_equal(rref_A, mat_mul(M, A))
            np.testing.assert_array_equal(A, mat_mul(Mi, rref_A))
            np.testing.assert_array_equal(np.identity(7), mat_mul(Mi, M))
