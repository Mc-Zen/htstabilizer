from src.htstabilizer.lc_classes import *
from src.htstabilizer.graph import Graph

import unittest
import numpy as np


class TestLCClasses(unittest.TestCase):

    def test_class_size(self):
        self.assertEqual(LCClass2.count(), 2)
        self.assertEqual(LCClass3.count(), 5)
        self.assertEqual(LCClass4.count(), 18)
        self.assertEqual(LCClass5.count(), 93)

    def test_class_num_qubits(self):
        self.assertEqual(LCClass2(0).num_qubits(), 2)
        self.assertEqual(LCClass3(0).num_qubits(), 3)
        self.assertEqual(LCClass4(0).num_qubits(), 4)
        self.assertEqual(LCClass5(0).num_qubits(), 5)

    def test_determine_lc_class2(self):
        lc_class = determine_lc_class2(Stabilizer(["XY", "XI"]))
        self.assertEqual(lc_class.type, LCClass2.EntanglementStructure.Separable)
        self.assertEqual(lc_class.id(), 0)
        self.assertEqual(lc_class, LCClass2(lc_class.id()))
        lc_class.get_graph()

        lc_class = determine_lc_class2(Stabilizer(["XY", "ZX"]))
        self.assertEqual(lc_class.type, LCClass2.EntanglementStructure.Entangled)
        self.assertEqual(lc_class.id(), 1)
        self.assertEqual(lc_class, LCClass2(lc_class.id()))

    def test_determine_lc_class3(self):
        lc_class = determine_lc_class3(Stabilizer(["XYI", "XIZ", "IYZ"]))
        self.assertEqual(lc_class.type, LCClass3.EntanglementStructure.Separable)
        self.assertEqual(lc_class.id(), 0)
        self.assertEqual(lc_class, LCClass3(lc_class.id()))

        lc_class = determine_lc_class3(Stabilizer(["YII", "IYX", "IXY"]))
        self.assertEqual(lc_class.type, LCClass3.EntanglementStructure.Pair)
        self.assertEqual(lc_class.data, linear_index.Repr([[0], [1, 2]]))
        self.assertEqual(lc_class.id(), 1)
        self.assertEqual(lc_class, LCClass3(lc_class.id()))

        lc_class = determine_lc_class3(Stabilizer(["XZI", "YIZ", "IYX"]))
        self.assertEqual(lc_class.type, LCClass3.EntanglementStructure.Triple)
        self.assertEqual(lc_class.id(), 4)
        self.assertEqual(lc_class, LCClass3(lc_class.id()))

    def test_determine_lc_class4(self):
        lc_class = determine_lc_class4(Stabilizer(["XYIY", "XIZY", "IYZI", "IYII"]))
        self.assertEqual(lc_class.type, LCClass4.EntanglementStructure.Separable)
        self.assertEqual(lc_class.id(), 0)
        self.assertEqual(lc_class, LCClass4(lc_class.id()))

        lc_class = determine_lc_class4(Stabilizer(["YIIY", "IYXY", "IXYI", "YXYI"]))
        self.assertEqual(lc_class.type, LCClass4.EntanglementStructure.Pair)
        self.assertEqual(lc_class.data, linear_index.Repr([[0], [3], [1, 2]]))
        self.assertEqual(lc_class.id(), 4)
        self.assertEqual(lc_class.data, LCClass4(lc_class.id()).data)

        lc_class = determine_lc_class4(Stabilizer(["YXYX", "XZII", "ZYII", "YXZY"]))
        self.assertEqual(lc_class.type, LCClass4.EntanglementStructure.TwoPairs)
        self.assertEqual(lc_class.data, linear_index.Repr([[0, 1], [2, 3]]))
        self.assertEqual(lc_class.id(), 11)
        self.assertEqual(lc_class, LCClass4(lc_class.id()))

        lc_class = determine_lc_class4(Stabilizer(["XZIY", "YIZY", "IYXI", "XYZY"]))
        self.assertEqual(lc_class.type, LCClass4.EntanglementStructure.Triple)
        self.assertEqual(lc_class.data, linear_index.Repr([[0, 1, 2], [3]]))
        self.assertEqual(lc_class.id(), 10)
        self.assertEqual(lc_class, LCClass4(lc_class.id()))

        lc_class = determine_lc_class4(Stabilizer(["XYYY", "YIZI", "IYII", "YYIX"]))
        self.assertEqual(lc_class.type, LCClass4.EntanglementStructure.Triple)
        self.assertEqual(lc_class.data, linear_index.Repr([[0, 2, 3], [1]]))
        self.assertEqual(lc_class.id(), 8)
        self.assertEqual(lc_class, LCClass4(lc_class.id()))

        lc_class = determine_lc_class4(Stabilizer(["XYXX", "YZII", "ZXZZ", "XXIY"]))
        self.assertEqual(lc_class.type, LCClass4.EntanglementStructure.Line)
        self.assertEqual(lc_class.data, linear_index.Repr([[0, 1], [2, 3]]))
        self.assertEqual(lc_class.id(), 15)
        self.assertEqual(lc_class, LCClass4(lc_class.id()))

        lc_class = determine_lc_class4(Stabilizer(["YYXX", "YIZZ", "IXIZ", "XYIY"]))
        self.assertEqual(lc_class.type, LCClass4.EntanglementStructure.Line)
        self.assertEqual(lc_class.data, linear_index.Repr([[0, 2], [1, 3]]))
        self.assertEqual(lc_class.id(), 16)
        self.assertEqual(lc_class, LCClass4(lc_class.id()))

        lc_class = determine_lc_class4(Stabilizer(["YXXX", "XZZZ", "ZXZI", "XYIX"]))
        self.assertEqual(lc_class.type, LCClass4.EntanglementStructure.Line)
        self.assertEqual(lc_class.data, linear_index.Repr([[0, 3], [1, 2]]))
        self.assertEqual(lc_class.id(), 17)
        self.assertEqual(lc_class, LCClass4(lc_class.id()))

        lc_class = determine_lc_class4(Stabilizer(["ZXYX", "ZIIX", "XYZY", "ZXII"]))
        self.assertEqual(lc_class.type, LCClass4.EntanglementStructure.Star)
        self.assertEqual(lc_class.id(), 14)
        self.assertEqual(lc_class, LCClass4(lc_class.id()))

    def test_determine_lc_class5(self):
        g = Graph(5)
        lc_class = determine_lc_class5(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass5.EntanglementStructure.Separable)
        self.assertEqual(lc_class.id(), 0)
        self.assertEqual(lc_class, LCClass5(lc_class.id()))

        g = Graph(5)
        g.add_edge(3, 0)
        lc_class = determine_lc_class5(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass5.EntanglementStructure.Pair)
        self.assertEqual(lc_class.data, linear_index.Repr([[0, 3], [1, 2, 4]]))
        self.assertEqual(lc_class.id(), 3)
        self.assertEqual(lc_class, LCClass5(lc_class.id()))

        g = Graph(5)
        g.add_edge(3, 0)
        g.add_edge(2, 0)
        lc_class = determine_lc_class5(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass5.EntanglementStructure.Triple)
        self.assertEqual(lc_class.data, linear_index.Repr([[0, 2, 3], [1, 4]]))
        self.assertEqual(lc_class.id(), 6+11)
        self.assertEqual(lc_class, LCClass5(lc_class.id()))

        g = Graph(5)
        g.add_edge(3, 0)
        g.add_edge(2, 0)
        g.add_edge(2, 1)
        lc_class = determine_lc_class5(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass5.EntanglementStructure.Line4)
        self.assertEqual(lc_class.data, linear_index.Repr([[0, 3], [1, 2], [4]]))
        self.assertEqual(lc_class.id(), 55)
        self.assertEqual(lc_class, LCClass5(lc_class.id()))

        ###

        g = Graph(5)
        g.add_path([3, 0, 2, 1, 4])
        lc_class = determine_lc_class5(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass5.EntanglementStructure.Line)
        self.assertEqual(lc_class.data, linear_index.Repr([[0, 3], [1, 4], [2]]))
        self.assertEqual(lc_class.id(), 84)
        self.assertEqual(lc_class, LCClass5(lc_class.id()))

        g = Graph(5)
        g.add_edge(3, 0)
        g.add_edge(1, 4)
        lc_class = determine_lc_class5(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass5.EntanglementStructure.TwoPairs)
        self.assertEqual(lc_class.data, linear_index.Repr([[0, 3], [1, 4], [2]]))
        self.assertEqual(lc_class.id(), 28)
        self.assertEqual(lc_class, LCClass5(lc_class.id()))

        g = Graph(5)
        g.add_edge(1, 4)
        g.add_edge(2, 3)
        lc_class = determine_lc_class5(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass5.EntanglementStructure.TwoPairs)
        self.assertEqual(lc_class.data, linear_index.Repr([[1, 4], [2, 3], [0]]))
        self.assertEqual(lc_class.id(), 23)
        self.assertEqual(lc_class, LCClass5(lc_class.id()))

        g = Graph.star(5)
        lc_class = determine_lc_class5(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass5.EntanglementStructure.Star)
        self.assertEqual(lc_class.id(), 56)
        self.assertEqual(lc_class, LCClass5(lc_class.id()))

        g = Graph.star(5)
        g.remove_edge(0, 3)
        lc_class = determine_lc_class5(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass5.EntanglementStructure.Star4)
        self.assertEqual(lc_class.data, linear_index.Repr([[3], [0, 1, 2, 4]]))
        self.assertEqual(lc_class.id(), 36+3)
        self.assertEqual(lc_class, LCClass5(lc_class.id()))

        g = Graph(5)
        g.add_path([1, 4, 3])
        g.add_edge(0, 2)
        lc_class = determine_lc_class5(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass5.EntanglementStructure.PairAndTriple)
        self.assertEqual(lc_class.data, linear_index.Repr([[0, 2], [1, 3, 4]]))
        self.assertEqual(lc_class.id(), 58)
        self.assertEqual(lc_class, LCClass5(lc_class.id()))

        g = Graph(5)
        g.add_path([1, 4, 3, 0, 2, 1])
        g.add_edge(0, 2)
        lc_class = determine_lc_class5(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass5.EntanglementStructure.Cycle)
        self.assertEqual(lc_class.id(), 92)
        self.assertEqual(lc_class, LCClass5(lc_class.id()))

        g = Graph(5)
        g.add_path([1, 4, 3, 0])
        g.add_edge(3, 2)
        lc_class = determine_lc_class5(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass5.EntanglementStructure.T)
        self.assertEqual(lc_class.data, linear_index.Repr([[1, 4], [0, 2, 3]]))
        self.assertEqual(lc_class.id(), 73)
        self.assertEqual(lc_class, LCClass5(lc_class.id()))

    def assert_equal_class(self, lc_class, sort=False):
        cls = type(lc_class)
        new_lc_class = cls(lc_class.id())

        if sort:
            for group in new_lc_class.data.groups:
                group.sort(key=lambda x: x.data[0])
            for group in lc_class.data.groups:
                group.sort(key=lambda x: x.data[0])
        self.assertEqual(lc_class, new_lc_class)

    def test_determine_lc_class6(self):
        g = Graph(6)
        g.add_path([3, 4, 0, 5])
        lc_class = determine_lc_class6(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass6.EntanglementStructure.Line4)
        self.assertEqual(lc_class.data, linear_index.Repr([[3, 4], [0, 5], [1], [2]]))
        self.assert_equal_class(lc_class, sort=True)

        g = Graph(6)
        g.add_path([1, 0, 2, 4, 5, 1])
        lc_class = determine_lc_class6(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass6.EntanglementStructure.Cycle5)
        self.assertEqual(lc_class.data, linear_index.Repr([[3], [0, 1, 2, 4, 5]]))
        self.assert_equal_class(lc_class, sort=True)

        g = Graph(6)
        g.add_path([3, 4, 1, 2, 5])
        lc_class = determine_lc_class6(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass6.EntanglementStructure.Line5)
        self.assertEqual(lc_class.data, linear_index.Repr([[0], [1], [3, 4], [2, 5]]))
        self.assert_equal_class(lc_class, sort=True)

        g = Graph(6)
        g.add_path([3, 4, 0])
        g.add_edge(1, 4)
        lc_class = determine_lc_class6(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass6.EntanglementStructure.Star4)
        self.assertEqual(lc_class.data, linear_index.Repr([[0, 1, 3, 4], [2, 5]]))
        self.assert_equal_class(lc_class, sort=True)

        g = Graph(6)
        g.add_edge(1, 4)
        g.add_edge(2, 5)
        lc_class = determine_lc_class6(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass6.EntanglementStructure.TwoPairs)
        self.assertEqual(lc_class.data, linear_index.Repr([[0], [3], [1, 4], [2, 5]]))
        self.assert_equal_class(lc_class, sort=True)

        g = Graph(6)
        g.add_edge(1, 4)
        g.add_edge(3, 5)
        g.add_edge(2, 0)
        lc_class = determine_lc_class6(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass6.EntanglementStructure.ThreePairs)
        self.assertEqual(lc_class.data, linear_index.Repr([[0, 2], [1, 4], [3, 5]]))
        self.assert_equal_class(lc_class, sort=True)

        g = Graph(6)
        g.add_path([3, 4, 0])
        g.add_edge(1, 4)
        g.add_edge(2, 5)
        lc_class = determine_lc_class6(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass6.EntanglementStructure.Star4AndPair)
        self.assertEqual(lc_class.data, linear_index.Repr([[0, 1, 3, 4], [2, 5]]))
        self.assert_equal_class(lc_class, sort=True)

        g = Graph(6)
        g.add_path([3, 4, 0])
        g.add_path([1, 5, 2])
        lc_class = determine_lc_class6(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass6.EntanglementStructure.TwoTriples)
        self.assertEqual(lc_class.data, linear_index.Repr([[0, 3, 4], [1, 2, 5]]))
        self.assert_equal_class(lc_class, sort=True)

        g = Graph(6)
        g.add_path([1, 0, 2])
        g.add_path([4, 0, 5])
        lc_class = determine_lc_class6(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass6.EntanglementStructure.Star5)
        self.assertEqual(lc_class.data, linear_index.Repr([[3], [0, 1, 2, 4, 5]]))
        self.assert_equal_class(lc_class, sort=True)

        g = Graph(6)
        g.add_path([3, 4, 0, 2])
        g.add_path([1, 5])
        lc_class = determine_lc_class6(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass6.EntanglementStructure.Line4AndPair)
        self.assertEqual(lc_class.data, linear_index.Repr([[1], [5], [0, 2], [3, 4]]))
        self.assert_equal_class(lc_class, sort=True)

        g = Graph(6)
        g.add_path([2, 1, 0, 4, 3, 5, 2])
        lc_class = determine_lc_class6(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass6.EntanglementStructure.Cycle6)
        self.assertEqual(lc_class.data, linear_index.Repr([[1, 3], [2, 4], [0, 5]]))
        self.assert_equal_class(lc_class, sort=True)

        g = Graph(6)
        g.add_path([3, 4, 0, 2])
        g.add_path([0, 1])
        lc_class = determine_lc_class6(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass6.EntanglementStructure.T)
        self.assertEqual(lc_class.data, linear_index.Repr([[5], [0, 1, 2], [3, 4]]))
        self.assert_equal_class(lc_class, sort=True)

        g = Graph(6)
        g.add_path([2, 1, 0, 4, 3, 5])
        lc_class = determine_lc_class6(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass6.EntanglementStructure.Line6)
        self.assertEqual(lc_class.data, linear_index.Repr([[0], [4], [1, 2], [3, 5]]))
        self.assert_equal_class(lc_class, sort=True)

        g = Graph(6)
        g.add_path([3, 4, 0])
        g.add_path([2, 5])
        lc_class = determine_lc_class6(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass6.EntanglementStructure.PairAndTriple)
        self.assertEqual(lc_class.data, linear_index.Repr([[1], [2, 5], [0, 3, 4]]))
        self.assert_equal_class(lc_class, sort=True)

        g = Graph(6)
        g.add_path([2, 1, 0, 4, 3])
        g.add_path([4, 5])
        lc_class = determine_lc_class6(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass6.EntanglementStructure.T6)
        self.assertEqual(lc_class.data, linear_index.Repr([[1, 2], [0], [3, 4, 5]]))
        self.assert_equal_class(lc_class, sort=True)

        g = Graph(6)
        g.add_path([2, 1, 0, 4])
        g.add_path([0, 5])
        g.add_path([0, 3])
        lc_class = determine_lc_class6(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass6.EntanglementStructure.Cross)
        self.assertEqual(lc_class.data, linear_index.Repr([[1, 2], [0, 3, 4, 5]]))
        self.assert_equal_class(lc_class, sort=True)

        g = Graph(6)
        g.add_path([0, 1, 2])
        g.add_path([3, 1, 4])
        g.add_path([5, 1])
        lc_class = determine_lc_class6(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass6.EntanglementStructure.Star6)
        self.assert_equal_class(lc_class, sort=True)

        g = Graph(6)
        g.add_path([3, 4, 0])
        g.add_path([1, 5, 2])
        g.add_path([4, 5])
        lc_class = determine_lc_class6(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass6.EntanglementStructure.H)
        self.assertEqual(lc_class.data, linear_index.Repr([[1, 2, 5], [0, 3, 4]]))
        self.assert_equal_class(lc_class, sort=True)

        g = Graph(6)
        g.add_path([2, 1, 0, 4, 5])
        g.add_path([0, 3])
        lc_class = determine_lc_class6(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass6.EntanglementStructure.E)
        self.assertEqual(lc_class.data, linear_index.Repr([[0], [3], [1, 2], [4, 5]]))
        self.assert_equal_class(lc_class, sort=True)

        g = Graph(6)
        g.add_path([2, 1, 0, 4, 2])
        g.add_path([2, 3])
        g.add_path([0, 5])
        lc_class = determine_lc_class6(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass6.EntanglementStructure.Box4)
        self.assertEqual(lc_class.data, linear_index.Repr([[2, 3], [1, 4], [0, 5]]))
        self.assert_equal_class(lc_class, sort=True)

        g = Graph(6)
        g.add_path([2, 1, 0, 4, 5])
        g.add_path([0, 3])
        g.add_path([1, 4])
        lc_class = determine_lc_class6(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass6.EntanglementStructure.EBar)
        self.assertEqual(lc_class.data, linear_index.Repr([[1, 2], [0, 3], [4, 5]]))
        self.assert_equal_class(lc_class, sort=True)

        g = Graph(6)
        g.add_path([2, 1, 0, 4, 3, 2])
        g.add_path([0, 5])
        lc_class = determine_lc_class6(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass6.EntanglementStructure.Box5)
        self.assertEqual(lc_class.data, linear_index.Repr([[0], [5], [1, 2], [3, 4]]))
        self.assert_equal_class(lc_class, sort=True)

        g = Graph(6)
        g.add_path([2, 1, 0, 4, 3, 5, 2])
        g.add_path([2, 0])
        g.add_path([4, 5])
        g.add_path([1, 3])
        lc_class = determine_lc_class6(Stabilizer(g))
        self.assertEqual(lc_class.type, LCClass6.EntanglementStructure.AME)
        self.assert_equal_class(lc_class, sort=True)

    def test_6_qubit_classes_to_from_graph(self):
        for i in range(LCClass6.count()):
            self.assertEqual(i, LCClass6(i).id())
            if i != determine_lc_class(Stabilizer(LCClass6(i).get_graph())).id():
                # LCClass6(i).get_graph().draw(show=True)
                print(LCClass6.get_LC_type(i))
            self.assertEqual(i, determine_lc_class(Stabilizer(LCClass6(i).get_graph())).id())

    def test_get_type(self):
        self.assertEqual(LCClass2.get_entanglement_structure(0), LCClass2.EntanglementStructure.Separable)
        self.assertEqual(LCClass2.get_entanglement_structure(1), LCClass2.EntanglementStructure.Entangled)

    def test_str(self):
        print(str(LCClass4(5)))

    def test_eq(self):
        self.assertEqual(LCClass4(5), LCClass4(5))