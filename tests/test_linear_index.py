from src.htstabilizer.linear_index import *
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

    def test_to_from_123(self):
        self.assertEqual(to_123(0), Repr([NTuple([0]), NTuple([1, 2]), NTuple([3, 4, 5])]))
        self.assertEqual(to_123(1), Repr([NTuple([0]), NTuple([1, 3]), NTuple([2, 4, 5])]))
        self.assertEqual(to_123(2), Repr([NTuple([0]), NTuple([1, 4]), NTuple([2, 3, 5])]))
        self.assertEqual(to_123(3), Repr([NTuple([0]), NTuple([1, 5]), NTuple([2, 3, 4])]))
        self.assertEqual(to_123(4), Repr([NTuple([0]), NTuple([2, 3]), NTuple([1, 4, 5])]))
        self.assertEqual(to_123(5), Repr([NTuple([0]), NTuple([2, 4]), NTuple([1, 3, 5])]))
        self.assertEqual(to_123(6), Repr([NTuple([0]), NTuple([2, 5]), NTuple([1, 3, 4])]))
        self.assertEqual(to_123(7), Repr([NTuple([0]), NTuple([3, 4]), NTuple([1, 2, 5])]))
        self.assertEqual(to_123(8), Repr([NTuple([0]), NTuple([3, 5]), NTuple([1, 2, 4])]))
        self.assertEqual(to_123(9), Repr([NTuple([0]), NTuple([4, 5]), NTuple([1, 2, 3])]))
        self.assertEqual(to_123(10), Repr([NTuple([1]), NTuple([2, 3]), NTuple([0, 4, 5])]))
        self.assertEqual(to_123(11), Repr([NTuple([1]), NTuple([2, 4]), NTuple([0, 3, 5])]))
        self.assertEqual(to_123(12), Repr([NTuple([1]), NTuple([2, 5]), NTuple([0, 3, 4])]))
        self.assertEqual(to_123(13), Repr([NTuple([1]), NTuple([0, 2]), NTuple([3, 4, 5])]))
        self.assertEqual(to_123(14), Repr([NTuple([1]), NTuple([3, 4]), NTuple([0, 2, 5])]))
        self.assertEqual(to_123(15), Repr([NTuple([1]), NTuple([3, 5]), NTuple([0, 2, 4])]))
        self.assertEqual(to_123(16), Repr([NTuple([1]), NTuple([0, 3]), NTuple([2, 4, 5])]))
        self.assertEqual(to_123(17), Repr([NTuple([1]), NTuple([4, 5]), NTuple([0, 2, 3])]))
        self.assertEqual(to_123(18), Repr([NTuple([1]), NTuple([0, 4]), NTuple([2, 3, 5])]))
        self.assertEqual(to_123(19), Repr([NTuple([1]), NTuple([0, 5]), NTuple([2, 3, 4])]))
        self.assertEqual(to_123(20), Repr([NTuple([2]), NTuple([3, 4]), NTuple([0, 1, 5])]))
        self.assertEqual(to_123(21), Repr([NTuple([2]), NTuple([3, 5]), NTuple([0, 1, 4])]))
        self.assertEqual(to_123(22), Repr([NTuple([2]), NTuple([0, 3]), NTuple([1, 4, 5])]))
        self.assertEqual(to_123(23), Repr([NTuple([2]), NTuple([1, 3]), NTuple([0, 4, 5])]))
        self.assertEqual(to_123(24), Repr([NTuple([2]), NTuple([4, 5]), NTuple([0, 1, 3])]))
        self.assertEqual(to_123(25), Repr([NTuple([2]), NTuple([0, 4]), NTuple([1, 3, 5])]))
        self.assertEqual(to_123(26), Repr([NTuple([2]), NTuple([1, 4]), NTuple([0, 3, 5])]))
        self.assertEqual(to_123(27), Repr([NTuple([2]), NTuple([0, 5]), NTuple([1, 3, 4])]))
        self.assertEqual(to_123(28), Repr([NTuple([2]), NTuple([1, 5]), NTuple([0, 3, 4])]))
        self.assertEqual(to_123(29), Repr([NTuple([2]), NTuple([0, 1]), NTuple([3, 4, 5])]))
        self.assertEqual(to_123(30), Repr([NTuple([3]), NTuple([4, 5]), NTuple([0, 1, 2])]))
        self.assertEqual(to_123(31), Repr([NTuple([3]), NTuple([0, 4]), NTuple([1, 2, 5])]))
        self.assertEqual(to_123(32), Repr([NTuple([3]), NTuple([1, 4]), NTuple([0, 2, 5])]))
        self.assertEqual(to_123(33), Repr([NTuple([3]), NTuple([2, 4]), NTuple([0, 1, 5])]))
        self.assertEqual(to_123(34), Repr([NTuple([3]), NTuple([0, 5]), NTuple([1, 2, 4])]))
        self.assertEqual(to_123(35), Repr([NTuple([3]), NTuple([1, 5]), NTuple([0, 2, 4])]))
        self.assertEqual(to_123(36), Repr([NTuple([3]), NTuple([2, 5]), NTuple([0, 1, 4])]))
        self.assertEqual(to_123(37), Repr([NTuple([3]), NTuple([0, 1]), NTuple([2, 4, 5])]))
        self.assertEqual(to_123(38), Repr([NTuple([3]), NTuple([0, 2]), NTuple([1, 4, 5])]))
        self.assertEqual(to_123(39), Repr([NTuple([3]), NTuple([1, 2]), NTuple([0, 4, 5])]))
        self.assertEqual(to_123(40), Repr([NTuple([4]), NTuple([0, 5]), NTuple([1, 2, 3])]))
        self.assertEqual(to_123(41), Repr([NTuple([4]), NTuple([1, 5]), NTuple([0, 2, 3])]))
        self.assertEqual(to_123(42), Repr([NTuple([4]), NTuple([2, 5]), NTuple([0, 1, 3])]))
        self.assertEqual(to_123(43), Repr([NTuple([4]), NTuple([3, 5]), NTuple([0, 1, 2])]))
        self.assertEqual(to_123(44), Repr([NTuple([4]), NTuple([0, 1]), NTuple([2, 3, 5])]))
        self.assertEqual(to_123(45), Repr([NTuple([4]), NTuple([0, 2]), NTuple([1, 3, 5])]))
        self.assertEqual(to_123(46), Repr([NTuple([4]), NTuple([0, 3]), NTuple([1, 2, 5])]))
        self.assertEqual(to_123(47), Repr([NTuple([4]), NTuple([1, 2]), NTuple([0, 3, 5])]))
        self.assertEqual(to_123(48), Repr([NTuple([4]), NTuple([1, 3]), NTuple([0, 2, 5])]))
        self.assertEqual(to_123(49), Repr([NTuple([4]), NTuple([2, 3]), NTuple([0, 1, 5])]))
        self.assertEqual(to_123(50), Repr([NTuple([5]), NTuple([0, 1]), NTuple([2, 3, 4])]))
        self.assertEqual(to_123(51), Repr([NTuple([5]), NTuple([0, 2]), NTuple([1, 3, 4])]))
        self.assertEqual(to_123(52), Repr([NTuple([5]), NTuple([0, 3]), NTuple([1, 2, 4])]))
        self.assertEqual(to_123(53), Repr([NTuple([5]), NTuple([0, 4]), NTuple([1, 2, 3])]))
        self.assertEqual(to_123(54), Repr([NTuple([5]), NTuple([1, 2]), NTuple([0, 3, 4])]))
        self.assertEqual(to_123(55), Repr([NTuple([5]), NTuple([1, 3]), NTuple([0, 2, 4])]))
        self.assertEqual(to_123(56), Repr([NTuple([5]), NTuple([1, 4]), NTuple([0, 2, 3])]))
        self.assertEqual(to_123(57), Repr([NTuple([5]), NTuple([2, 3]), NTuple([0, 1, 4])]))
        self.assertEqual(to_123(58), Repr([NTuple([5]), NTuple([2, 4]), NTuple([0, 1, 3])]))
        self.assertEqual(to_123(59), Repr([NTuple([5]), NTuple([3, 4]), NTuple([0, 1, 2])]))

        for i in range(60):
            self.assertEqual(from_123(to_123(i)), i)

    def test_to_from_33(self):
        self.assertEqual(to_33(0), Repr([NTuple([0, 1, 2]), NTuple([3, 4, 5])]))
        self.assertEqual(to_33(1), Repr([NTuple([0, 1, 3]), NTuple([2, 4, 5])]))
        self.assertEqual(to_33(2), Repr([NTuple([0, 1, 4]), NTuple([2, 3, 5])]))
        self.assertEqual(to_33(3), Repr([NTuple([0, 1, 5]), NTuple([2, 3, 4])]))
        self.assertEqual(to_33(4), Repr([NTuple([0, 2, 3]), NTuple([1, 4, 5])]))
        self.assertEqual(to_33(5), Repr([NTuple([0, 2, 4]), NTuple([1, 3, 5])]))
        self.assertEqual(to_33(6), Repr([NTuple([0, 2, 5]), NTuple([1, 3, 4])]))
        self.assertEqual(to_33(7), Repr([NTuple([0, 3, 4]), NTuple([1, 2, 5])]))
        self.assertEqual(to_33(8), Repr([NTuple([0, 3, 5]), NTuple([1, 2, 4])]))
        self.assertEqual(to_33(9), Repr([NTuple([0, 4, 5]), NTuple([1, 2, 3])]))

        for i in range(10):
            self.assertEqual(from_33(to_33(i)), i)

        self.assertEqual(from_33(Repr([NTuple([0, 3, 2]), NTuple([1, 4, 5])])), 4)
        self.assertEqual(from_33(Repr([NTuple([2, 3, 0]), NTuple([1, 4, 5])])), 4)
        self.assertEqual(from_33(Repr([NTuple([1, 4, 5]), NTuple([0, 3, 2])])), 4)
        self.assertEqual(from_33(Repr([NTuple([1, 4, 5]), NTuple([3, 0, 2])])), 4)

    def test_to_from_24(self):
        self.assertEqual(to_24(0), Repr([NTuple([0, 1]), NTuple([2, 3, 4, 5])]))
        self.assertEqual(to_24(1), Repr([NTuple([0, 2]), NTuple([1, 3, 4, 5])]))
        self.assertEqual(to_24(2), Repr([NTuple([0, 3]), NTuple([1, 2, 4, 5])]))
        self.assertEqual(to_24(3), Repr([NTuple([0, 4]), NTuple([1, 2, 3, 5])]))
        self.assertEqual(to_24(4), Repr([NTuple([0, 5]), NTuple([1, 2, 3, 4])]))
        self.assertEqual(to_24(5), Repr([NTuple([1, 2]), NTuple([0, 3, 4, 5])]))
        self.assertEqual(to_24(6), Repr([NTuple([1, 3]), NTuple([0, 2, 4, 5])]))
        self.assertEqual(to_24(7), Repr([NTuple([1, 4]), NTuple([0, 2, 3, 5])]))
        self.assertEqual(to_24(8), Repr([NTuple([1, 5]), NTuple([0, 2, 3, 4])]))
        self.assertEqual(to_24(9), Repr([NTuple([2, 3]), NTuple([0, 1, 4, 5])]))
        self.assertEqual(to_24(10), Repr([NTuple([2, 4]), NTuple([0, 1, 3, 5])]))
        self.assertEqual(to_24(11), Repr([NTuple([2, 5]), NTuple([0, 1, 3, 4])]))
        self.assertEqual(to_24(12), Repr([NTuple([3, 4]), NTuple([0, 1, 2, 5])]))
        self.assertEqual(to_24(13), Repr([NTuple([3, 5]), NTuple([0, 1, 2, 4])]))
        self.assertEqual(to_24(14), Repr([NTuple([4, 5]), NTuple([0, 1, 2, 3])]))

        for i in range(15):
            self.assertEqual(from_24(to_24(i)), i)

        self.assertEqual(from_24(Repr([NTuple([5, 0]), NTuple([1, 2, 3, 4])])), 4)

    def test_to_from_222(self):
        print()
        for i in range(15):
            print(to_222(i))
        self.assertEqual(to_222(0), Repr([NTuple([0, 1]), NTuple([2, 3]), NTuple([4, 5])]))
        self.assertEqual(to_222(1), Repr([NTuple([0, 1]), NTuple([2, 4]), NTuple([3, 5])]))
        self.assertEqual(to_222(2), Repr([NTuple([0, 1]), NTuple([2, 5]), NTuple([3, 4])]))
        self.assertEqual(to_222(3), Repr([NTuple([0, 2]), NTuple([1, 3]), NTuple([4, 5])]))
        self.assertEqual(to_222(4), Repr([NTuple([0, 2]), NTuple([1, 4]), NTuple([3, 5])]))
        self.assertEqual(to_222(5), Repr([NTuple([0, 2]), NTuple([1, 5]), NTuple([3, 4])]))
        self.assertEqual(to_222(6), Repr([NTuple([0, 3]), NTuple([1, 2]), NTuple([4, 5])]))
        self.assertEqual(to_222(7), Repr([NTuple([0, 3]), NTuple([1, 4]), NTuple([2, 5])]))
        self.assertEqual(to_222(8), Repr([NTuple([0, 3]), NTuple([1, 5]), NTuple([2, 4])]))
        self.assertEqual(to_222(9), Repr([NTuple([0, 4]), NTuple([1, 2]), NTuple([3, 5])]))
        self.assertEqual(to_222(10), Repr([NTuple([0, 4]), NTuple([1, 3]), NTuple([2, 5])]))
        self.assertEqual(to_222(11), Repr([NTuple([0, 4]), NTuple([1, 5]), NTuple([2, 3])]))
        self.assertEqual(to_222(12), Repr([NTuple([0, 5]), NTuple([1, 2]), NTuple([3, 4])]))
        self.assertEqual(to_222(13), Repr([NTuple([0, 5]), NTuple([1, 3]), NTuple([2, 4])]))
        self.assertEqual(to_222(14), Repr([NTuple([0, 5]), NTuple([1, 4]), NTuple([2, 3])]))

        for i in range(15):
            self.assertEqual(from_222(to_222(i)), i)

        self.assertEqual(from_222(Repr([NTuple([5, 0]), NTuple([3, 4]), NTuple([2, 1])])), 12)
        self.assertEqual(from_222(Repr([NTuple([3, 4]), NTuple([5, 0]), NTuple([2, 1])])), 12)
        self.assertEqual(from_222(Repr([NTuple([5, 4]), NTuple([1, 0]), NTuple([2, 3])])), 0)

    def test_to_from_1122(self):
        print()
        for i in range(45):
            self.assertEqual(sorted(to_1122(i).flatten()), list(range(6)))
            print(to_1122(i))

        self.assertEqual(to_1122(0), Repr([NTuple([0]), NTuple([1]), NTuple([2, 3]), NTuple([4, 5])]))
        self.assertEqual(to_1122(1), Repr([NTuple([0]), NTuple([1]), NTuple([2, 4]), NTuple([3, 5])]))
        self.assertEqual(to_1122(2), Repr([NTuple([0]), NTuple([1]), NTuple([2, 5]), NTuple([3, 4])]))
        self.assertEqual(to_1122(3), Repr([NTuple([0]), NTuple([2]), NTuple([1, 3]), NTuple([4, 5])]))
        self.assertEqual(to_1122(4), Repr([NTuple([0]), NTuple([2]), NTuple([1, 4]), NTuple([3, 5])]))
        self.assertEqual(to_1122(5), Repr([NTuple([0]), NTuple([2]), NTuple([1, 5]), NTuple([3, 4])]))
        self.assertEqual(to_1122(6), Repr([NTuple([0]), NTuple([3]), NTuple([1, 2]), NTuple([4, 5])]))
        self.assertEqual(to_1122(7), Repr([NTuple([0]), NTuple([3]), NTuple([1, 4]), NTuple([2, 5])]))
        self.assertEqual(to_1122(8), Repr([NTuple([0]), NTuple([3]), NTuple([1, 5]), NTuple([2, 4])]))
        self.assertEqual(to_1122(9), Repr([NTuple([0]), NTuple([4]), NTuple([1, 2]), NTuple([3, 5])]))
        self.assertEqual(to_1122(10), Repr([NTuple([0]), NTuple([4]), NTuple([1, 3]), NTuple([2, 5])]))
        self.assertEqual(to_1122(11), Repr([NTuple([0]), NTuple([4]), NTuple([1, 5]), NTuple([2, 3])]))
        self.assertEqual(to_1122(12), Repr([NTuple([0]), NTuple([5]), NTuple([1, 2]), NTuple([3, 4])]))
        self.assertEqual(to_1122(13), Repr([NTuple([0]), NTuple([5]), NTuple([1, 3]), NTuple([2, 4])]))
        self.assertEqual(to_1122(14), Repr([NTuple([0]), NTuple([5]), NTuple([1, 4]), NTuple([2, 3])]))
        self.assertEqual(to_1122(15), Repr([NTuple([1]), NTuple([2]), NTuple([0, 3]), NTuple([4, 5])]))
        self.assertEqual(to_1122(16), Repr([NTuple([1]), NTuple([2]), NTuple([0, 4]), NTuple([3, 5])]))
        self.assertEqual(to_1122(17), Repr([NTuple([1]), NTuple([2]), NTuple([0, 5]), NTuple([3, 4])]))
        self.assertEqual(to_1122(18), Repr([NTuple([1]), NTuple([3]), NTuple([0, 2]), NTuple([4, 5])]))
        self.assertEqual(to_1122(19), Repr([NTuple([1]), NTuple([3]), NTuple([0, 4]), NTuple([2, 5])]))
        self.assertEqual(to_1122(20), Repr([NTuple([1]), NTuple([3]), NTuple([0, 5]), NTuple([2, 4])]))
        self.assertEqual(to_1122(21), Repr([NTuple([1]), NTuple([4]), NTuple([0, 2]), NTuple([3, 5])]))
        self.assertEqual(to_1122(22), Repr([NTuple([1]), NTuple([4]), NTuple([0, 3]), NTuple([2, 5])]))
        self.assertEqual(to_1122(23), Repr([NTuple([1]), NTuple([4]), NTuple([0, 5]), NTuple([2, 3])]))
        self.assertEqual(to_1122(24), Repr([NTuple([1]), NTuple([5]), NTuple([0, 2]), NTuple([3, 4])]))
        self.assertEqual(to_1122(25), Repr([NTuple([1]), NTuple([5]), NTuple([0, 3]), NTuple([2, 4])]))
        self.assertEqual(to_1122(26), Repr([NTuple([1]), NTuple([5]), NTuple([0, 4]), NTuple([2, 3])]))
        self.assertEqual(to_1122(27), Repr([NTuple([2]), NTuple([3]), NTuple([0, 1]), NTuple([4, 5])]))
        self.assertEqual(to_1122(28), Repr([NTuple([2]), NTuple([3]), NTuple([0, 4]), NTuple([1, 5])]))
        self.assertEqual(to_1122(29), Repr([NTuple([2]), NTuple([3]), NTuple([0, 5]), NTuple([1, 4])]))
        self.assertEqual(to_1122(30), Repr([NTuple([2]), NTuple([4]), NTuple([0, 1]), NTuple([3, 5])]))
        self.assertEqual(to_1122(31), Repr([NTuple([2]), NTuple([4]), NTuple([0, 3]), NTuple([1, 5])]))
        self.assertEqual(to_1122(32), Repr([NTuple([2]), NTuple([4]), NTuple([0, 5]), NTuple([1, 3])]))
        self.assertEqual(to_1122(33), Repr([NTuple([2]), NTuple([5]), NTuple([0, 1]), NTuple([3, 4])]))
        self.assertEqual(to_1122(34), Repr([NTuple([2]), NTuple([5]), NTuple([0, 3]), NTuple([1, 4])]))
        self.assertEqual(to_1122(35), Repr([NTuple([2]), NTuple([5]), NTuple([0, 4]), NTuple([1, 3])]))
        self.assertEqual(to_1122(36), Repr([NTuple([3]), NTuple([4]), NTuple([0, 1]), NTuple([2, 5])]))
        self.assertEqual(to_1122(37), Repr([NTuple([3]), NTuple([4]), NTuple([0, 2]), NTuple([1, 5])]))
        self.assertEqual(to_1122(38), Repr([NTuple([3]), NTuple([4]), NTuple([0, 5]), NTuple([1, 2])]))
        self.assertEqual(to_1122(39), Repr([NTuple([3]), NTuple([5]), NTuple([0, 1]), NTuple([2, 4])]))
        self.assertEqual(to_1122(40), Repr([NTuple([3]), NTuple([5]), NTuple([0, 2]), NTuple([1, 4])]))
        self.assertEqual(to_1122(41), Repr([NTuple([3]), NTuple([5]), NTuple([0, 4]), NTuple([1, 2])]))
        self.assertEqual(to_1122(42), Repr([NTuple([4]), NTuple([5]), NTuple([0, 1]), NTuple([2, 3])]))
        self.assertEqual(to_1122(43), Repr([NTuple([4]), NTuple([5]), NTuple([0, 2]), NTuple([1, 3])]))
        self.assertEqual(to_1122(44), Repr([NTuple([4]), NTuple([5]), NTuple([0, 3]), NTuple([1, 2])]))

        for i in range(15):
            self.assertEqual(from_1122(to_1122(i)), i)

        self.assertEqual(from_1122(Repr([NTuple([0]), NTuple([3]), NTuple([5, 1]), NTuple([2, 4])])), 8)
        self.assertEqual(from_1122(Repr([NTuple([0]), NTuple([3]), NTuple([5, 1]), NTuple([2, 4])])), 8)
        self.assertEqual(from_1122(Repr([NTuple([3]), NTuple([0]), NTuple([2, 4]), NTuple([5, 1])])), 8)
