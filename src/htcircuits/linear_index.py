""" 
Utilities to get a linear index for small groupings under certain symmetries
and also inversely create these groupings from the linear index. 
"""

from collections import defaultdict
from typing import List
from typing import Tuple
import numpy as np


class NTuple:
    """Class for tuples of integer that are sorted upon creation"""

    def __init__(self, data: List[int]):
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
        return f"NTuple{self.data}"


class Repr:
    def __init__(self, data: NTuple | List[NTuple] | List[List[int]] | List[int] | None = None):
        self.groups = defaultdict(list)
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
        return self.groups[size][index]

    def add(self, tuple: NTuple):
        self.groups[len(tuple)].append(tuple)

    def __eq__(self, other) -> bool:
        return self.groups == other.groups


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
    return Repr([NTuple([i, j]), NTuple(rest)])


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
    elif pair2[1] == ap1:
        b, c = pair2[1], pair2[0]
    elif pair2[1] == ap1:
        b, c = pair2[1], pair2[0]
    return 3 * single + (5 + c - b) % 5 - 1
