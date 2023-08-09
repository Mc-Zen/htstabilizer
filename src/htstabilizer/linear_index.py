""" 
Utilities to get a linear index for small groupings under certain symmetries
and also inversely create these groupings from the linear index. 
"""

from typing import List, Union
from typing import Tuple
import numpy as np


class NTuple:
    """Class for tuples of integer that are sorted upon creation"""
    __slots__ = ("data")

    def __init__(self, data: Union[List[int], int]):
        if isinstance(data, int):
            data = [data]
        self.data = data
        self.data.sort()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> int:
        return self.data[index]

    def __eq__(self, other) -> bool:
        return self.data == other.data

    def __str__(self) -> str:
        return str(self.data)

    def __repr__(self) -> str:
        return f"NTuple({self.data})"


class Repr:
    __slots__ = ("groups")

    def __init__(self, data: Union[NTuple, List[NTuple], List[List[int]], List[int], None] = None):
        self.groups = []
        if data is None:
            return
        if isinstance(data, NTuple):
            data = [data]
        if isinstance(data[0], list):
            for entry in data:
                self.add(NTuple(entry))
            return
        elif isinstance(data[0], int):
            self.add(NTuple(data))
            return
        for entry in data:
            self.add(entry)

    def get(self, size, index):
        return self.groups[size - 1][index]

    def add(self, tuple: NTuple):
        if len(self.groups) < len(tuple):
            for i in range(len(tuple) - len(self.groups)):
                self.groups.append([])
        self.groups[len(tuple) - 1].append(tuple)

    def flatten(self) -> List[int]:
        result = []
        for groups in self.groups:
            for group in groups:
                result += group
        return result

    def __eq__(self, other) -> bool:
        return self.groups == other.groups

    def __repr__(self) -> str:
        items = [item for sublist in self.groups for item in sublist]
        return f"Repr({items})"


def linear_index_from_n_choose_2(n: int, i: int, j: int) -> int:
    """Get a unique linear index for pairs of values (i,j), i < j based on triangle numbers. 

    Parameters
    ----------
    n : int
        Size
    i : int
        Smaller of the two numbers
    j : int
        Larger of the two numbers

    Returns
    -------
    int
        linear index
    """
    assert i < j, "i needs to be smaller than j"
    return n * (n - 1) // 2 - (n - i) * (n - i - 1) // 2 + j - i - 1


def linear_index_to_n_choose2_to(n: int, index: int) -> Tuple[int, int]:
    """Inverse operation to linear_index_from_n_choose_2(n, i, j) -> index. From a linear index
    get unique pair of two indices i < j (see triangular numbers). 

    Parameters
    ----------
    n : int
        Size
    index : int
        linear index

    Returns
    -------
    Tuple[int, int]
        Unique pair of indices i < j
    """
    i = int(n - 2 - np.floor(np.sqrt(4 * n * (n - 1) - 7 - 8 * index) / 2.0 - 0.5))
    return i, index + i + 1 - n * (n - 1) // 2 + (n - i) * (n - i - 1) // 2


def to_0(index: int) -> Repr:
    return Repr()


def from_0(repr: Repr) -> int:
    return 0


def to_1n(n: int, index: int) -> Repr:
    rest = list(filter(lambda el: el != index, range(n)))
    return Repr([NTuple([index]), NTuple(rest)])


def from_1n(n: int, repr: Repr) -> int:
    return repr.get(1, 0)[0]


def to_12(index: int) -> Repr:
    return to_1n(3, index)


def from_12(repr: Repr) -> int:
    return from_1n(3, repr)


def to_13(index: int) -> Repr:
    return to_1n(4, index)


def from_13(repr: Repr) -> int:
    return from_1n(4, repr)


def to_14(index: int) -> Repr:
    return to_1n(5, index)


def from_14(repr: Repr) -> int:
    return from_1n(5, repr)


def to_15(index: int) -> Repr:
    return to_1n(6, index)


def from_15(repr: Repr) -> int:
    return from_1n(6, repr)


def to_22(index: int) -> Repr:
    rest = list(filter(lambda el: el != index + 1, range(1, 4)))
    return Repr([NTuple([0, index+1]), NTuple(rest)])


def from_22(repr: Repr) -> int:
    pair1 = repr.get(2, 0)
    pair2 = repr.get(2, 1)
    if pair1[0] == 0:  # either pair[1] or pair[2] is 0.
        return pair1[1] - 1
    return pair2[1] - 1


def to_112(index: int) -> Repr:
    i, j = linear_index_to_n_choose2_to(4, index)
    rest = list(filter(lambda el: not el in [i, j], range(4)))
    return Repr([NTuple([i, j]), NTuple(rest[0]), NTuple(rest[1])])


def from_112(repr: Repr) -> int:
    pair = repr.get(2, 0)
    return linear_index_from_n_choose_2(4, *pair)


def to_23(index: int) -> Repr:
    i, j = linear_index_to_n_choose2_to(5, index)
    rest = list(filter(lambda el: not el in [i, j], range(5)))
    return Repr([NTuple([i, j]), NTuple(rest)])


def from_23(repr: Repr) -> int:
    pair = repr.get(2, 0)
    return linear_index_from_n_choose_2(5, *pair)


def to_122(index: int) -> Repr:
    a = index // 3
    b = (a + 1) % 5
    c = (a + 2 + index % 3) % 5
    rest = list(filter(lambda el: not el in [a, b, c], range(5)))
    return Repr([NTuple([a]), NTuple([b, c]), NTuple(rest)])


def from_122(repr: Repr) -> int:
    single = repr.get(1, 0)[0]
    pair1 = repr.get(2, 0)
    pair2 = repr.get(2, 1)
    # Find p1 that contains a+1 mod 5
    ap1 = (single + 1) % 5
    b, c = 0, 0
    if pair1[0] == ap1:
        b, c = pair1[0], pair1[1]
    elif pair1[1] == ap1:
        b, c = pair1[1], pair1[0]
    elif pair2[0] == ap1:
        b, c = pair2[0], pair2[1]
    elif pair2[1] == ap1:
        b, c = pair2[1], pair2[0]
    return 3 * single + (5 + c - b) % 5 - 1


def to_123(index: int) -> Repr:
    a = index // 10
    b, c = linear_index_to_n_choose2_to(5, index % 10)
    b = (b+a+1) % 6
    c = (c+a+1) % 6
    rest = list(filter(lambda el: not el in [a, b, c], range(6)))
    return Repr([NTuple([a]), NTuple([b, c]), NTuple(rest)])


def from_123(repr: Repr) -> int:
    a = repr.get(1, 0)[0]
    pair = repr.get(2, 0)
    b, c = (pair[0]-a+5) % 6, (pair[1]-a+5) % 6
    b, c = min([b, c]), max([b, c])
    index = linear_index_from_n_choose_2(5, b, c)
    return 10 * a + index


def to_33(index: int) -> Repr:
    a = 0
    b, c = linear_index_to_n_choose2_to(5, index)
    b = (b+1) % 6
    c = (c+1) % 6
    rest = list(filter(lambda el: not el in [a, b, c], range(6)))
    return Repr([NTuple([a, b, c]), NTuple(rest)])


def from_33(repr: Repr) -> int:
    # find the tuple that has the 0

    triple1, triple2 = repr.get(3, 0), repr.get(3, 1)
    if 0 in triple2:
        triple1 = triple2
    rest = list(filter(lambda el: el != 0, triple1))
    assert len(rest) == 2
    b, c = (rest[0]+5) % 6, (rest[1]+5) % 6
    if b > c:
        b, c = c, b
    return linear_index_from_n_choose_2(5, b, c)


def to_24(index: int) -> Repr:
    i, j = linear_index_to_n_choose2_to(6, index)
    rest = list(filter(lambda el: not el in [i, j], range(6)))
    return Repr([NTuple([i, j]), NTuple(rest)])


def from_24(repr: Repr) -> int:
    pair = repr.get(2, 0)
    return linear_index_from_n_choose_2(6, *pair)


def to_222(index: int) -> Repr:
    a = 0
    b = 1 + index // 3
    c, d = linear_index_to_n_choose2_to(4, index % 3)
    c = 1
    d = c + 1 + index % 3
    if d >= b:
        d += 1

    if c >= b:
        c += 1
    rest = list(filter(lambda el: not el in (a, b, c, d), range(6)))
    return Repr([NTuple([a, b]), NTuple([c, d]), NTuple(rest)])


def from_222(repr: Repr) -> int:
    pairs = [repr.get(2, 0).data.copy(), repr.get(2, 1).data.copy(), repr.get(2, 2).data.copy()]
    pairs.sort(key=lambda x: x[0])
    assert pairs[0][0] == 0
    b = pairs[0][1]
    for pair in pairs[1:]:
        if pair[0] > b:
            pair[0] -= 1
        if pair[1] > b:
            pair[1] -= 1
    return 3*(b - 1) + pairs[1][1] - pairs[1][0] - 1


def to_1122(index: int) -> Repr:
    a, b = linear_index_to_n_choose2_to(6, index // 3)
    rest = list(filter(lambda el: not el in (a, b), range(6)))
    c, d = rest[0], rest[1 + index % 3]
    restrest = list(filter(lambda x: not x in (c, d), rest))
    return Repr([NTuple(a), NTuple(b), NTuple([c, d]), NTuple(restrest)])


def from_1122(repr: Repr) -> int:
    ab = [repr.get(1, 0)[0], repr.get(1, 1)[0]]
    ab.sort()
    i = linear_index_from_n_choose_2(6, *ab)

    pairs = [repr.get(2, 0).data, repr.get(2, 1).data]
    pairs.sort(key=lambda x: x[0])
    rest = list(filter(lambda el: not el in ab, range(6)))
    c = rest.index(pairs[0][0])
    d = rest.index(pairs[0][1])
    return 3*i + d - c - 1


def to_1113(index: int) -> Repr:
    l3 = to_33(index // 2)
    triple1, triple2 = l3.get(3, 0), l3.get(3, 1)
    if index % 2 == 0:
        triple1, triple2 = triple2, triple1

    return Repr([triple1, *[NTuple(x) for x in triple2]])


def from_1113(repr: Repr) -> int:
    index = from_33(Repr([repr.get(3, 0), NTuple([repr.get(1, 0)[0], repr.get(1, 1)[0], repr.get(1, 2)[0]])]))
    # the zero is in the triple for odd numbers

    return 2*index + int(0 in repr.get(3, 0))

# Note: exceptionally, for this case the order of the NTuples matters!
# The first 1-Tuple (single) is different to the second one and they are
# not interchangeable.


def to_1122s(index: int) -> Repr:
    # the way this works is by using the 1122 combinatorics and
    # breaking the symmetry between the two 1-Tuples. to_1122()
    # always returns tuples arranged so that the first 1-Tuple is
    # the smaller one. Here we define the
    l1122 = to_1122(index // 2)
    pair1, pair2 = l1122.get(2, 0), l1122.get(2, 1)
    single1, single2 = l1122.get(1, 0), l1122.get(1, 1)

    if index % 2 == 1:
        single1, single2 = single2, single1
    return Repr([pair1, pair2, single1, single2])


def from_1122s(repr: Repr) -> int:
    index = from_1122(repr)
    return 2*index + int(repr.get(1, 0)[0] > repr.get(1, 1)[0])

# 4 Qubits:
#   Config  |  Size
# ----------|-------
#   1111=4  |   1
#   13      |   4
#   112     |   6
#   22      |   3

# 5 Qubits:
#   Config  |  Size
# ----------|-------
#   11111=5 |   1
#   14      |   5
#   122     |  15
#   23      |  10

# 6 Qubits:
#   Config  |  Size
# ----------|-------
#  111111=6 |*   1
#  15       |*   6  = (6 choose 1)
#  33       |*  10  = (6 choose 3) / 2
#  222      |*  15  = (6 choose 2) * (4 choose 2) / 6
#  11112=24 |*  15  = (6 choose 2)
#  1113     |*  20  = (6 choose 3)
#  1122     |*  45  = (6 choose 2) * (4 choose 2) / 2
#  123      |*  60  = (6 choose 2) * (4 choose 1) = (6 choose 1) * (5 choose 2)
#  1122s    |*  90  = (6 choose 2) * (4 choose 2)
