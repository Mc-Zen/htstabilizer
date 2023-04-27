from src.htcircuits.linear_index import *
import unittest


class TestLinearIndex(unittest.TestCase):

    def test_NTuple(self):
        self.assertEqual(NTuple([0, 1, 2, 3]).data, [0, 1, 2, 3])
        self.assertEqual(NTuple([3, 0, 1, 2]).data, [0, 1, 2, 3])
        self.assertEqual(len(NTuple([3, 0, 1, 2])), 4)

    def test_Repr(self):
        self.assertEqual(Repr(NTuple([2, 3])), Repr([NTuple([2, 3])]))
        self.assertEqual(Repr([[2, 3]]), Repr([NTuple([2, 3])]))
        self.assertEqual(Repr([[2, 3], [0, 1]]), Repr([NTuple([2, 3]), NTuple([0, 1])]))
        self.assertEqual(Repr([2, 3]), Repr([NTuple([2, 3])]))

    def test_to_from_12(self):
        self.assertEqual(to_12(0), Repr([NTuple([0]), NTuple([1, 2])]))
        self.assertEqual(to_12(1), Repr([NTuple([1]), NTuple([0, 2])]))
        self.assertEqual(to_12(2), Repr([NTuple([2]), NTuple([0, 1])]))

        for i in range(3):
            self.assertEqual(from_12(to_12(i)), i)

    def test_to_from_13(self):
        self.assertEqual(to_13(0), Repr([NTuple([0]), NTuple([1, 2, 3])]))
        self.assertEqual(to_13(1), Repr([NTuple([1]), NTuple([0, 2, 3])]))
        self.assertEqual(to_13(2), Repr([NTuple([2]), NTuple([0, 1, 3])]))
        self.assertEqual(to_13(3), Repr([NTuple([3]), NTuple([0, 1, 2])]))

        for i in range(4):
            self.assertEqual(from_13(to_13(i)), i)

    def test_to_from_22(self):
        self.assertEqual(to_22(0), Repr([NTuple([0, 1]), NTuple([2, 3])]))
        self.assertEqual(to_22(1), Repr([NTuple([0, 2]), NTuple([1, 3])]))
        self.assertEqual(to_22(2), Repr([NTuple([0, 3]), NTuple([1, 2])]))

        for i in range(3):
            self.assertEqual(from_22(to_22(i)), i)

    def test_to_from_14(self):
        self.assertEqual(to_14(0), Repr([NTuple([0]), NTuple([1, 2, 3, 4])]))
        self.assertEqual(to_14(1), Repr([NTuple([1]), NTuple([0, 2, 3, 4])]))
        self.assertEqual(to_14(2), Repr([NTuple([2]), NTuple([0, 1, 3, 4])]))
        self.assertEqual(to_14(3), Repr([NTuple([3]), NTuple([0, 1, 2, 4])]))
        self.assertEqual(to_14(4), Repr([NTuple([4]), NTuple([0, 1, 2, 3])]))

        for i in range(5):
            self.assertEqual(from_14(to_14(i)), i)

    def test_to_from_122(self):
        self.assertEqual(to_122(0), Repr([NTuple([0]), NTuple([1, 2]), NTuple([3, 4])]))
        self.assertEqual(to_122(1), Repr([NTuple([0]), NTuple([1, 3]), NTuple([2, 4])]))
        self.assertEqual(to_122(2), Repr([NTuple([0]), NTuple([1, 4]), NTuple([2, 3])]))
        self.assertEqual(to_122(3), Repr([NTuple([1]), NTuple([2, 3]), NTuple([0, 4])]))
        self.assertEqual(to_122(4), Repr([NTuple([1]), NTuple([2, 4]), NTuple([0, 3])]))
        self.assertEqual(to_122(5), Repr([NTuple([1]), NTuple([2, 0]), NTuple([3, 4])]))
        self.assertEqual(to_122(6), Repr([NTuple([2]), NTuple([3, 4]), NTuple([0, 1])]))
        self.assertEqual(to_122(7), Repr([NTuple([2]), NTuple([3, 0]), NTuple([1, 4])]))
        self.assertEqual(to_122(8), Repr([NTuple([2]), NTuple([3, 1]), NTuple([0, 4])]))
        self.assertEqual(to_122(9), Repr([NTuple([3]), NTuple([4, 0]), NTuple([1, 2])]))
        self.assertEqual(to_122(10), Repr([NTuple([3]), NTuple([4, 1]), NTuple([0, 2])]))
        self.assertEqual(to_122(11), Repr([NTuple([3]), NTuple([4, 2]), NTuple([0, 1])]))
        self.assertEqual(to_122(12), Repr([NTuple([4]), NTuple([0, 1]), NTuple([2, 3])]))
        self.assertEqual(to_122(13), Repr([NTuple([4]), NTuple([0, 2]), NTuple([1, 3])]))
        self.assertEqual(to_122(14), Repr([NTuple([4]), NTuple([0, 3]), NTuple([1, 2])]))

        for i in range(15):
            self.assertEqual(from_122(to_122(i)), i)

        # special cases where the input order is different
        self.assertEqual(from_122(Repr([[1], [0, 4], [2, 3]])), 3)
        self.assertEqual(from_122(Repr([[1], [0, 4], [2, 3]])), 3)

    def test_linear_index_from_n_choose_2(self):
        for n in range(6):
            for i in range(n*(n-1) // 2):
                self.assertEqual(linear_index_from_n_choose_2(n, *linear_index_to_n_choose2_to(n, i)), i)
