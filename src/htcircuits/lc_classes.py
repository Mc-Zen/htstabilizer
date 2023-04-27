"""
Tools for determining equivalence classes under local complementation
for stabilizer groups.
"""
from enum import IntEnum, unique
from typing import List
from .stabilizer import Stabilizer
import numpy as np
import itertools
from . import linear_index
from .graph import Graph
import abc


class LCClassBase(metaclass=abc.ABCMeta):
    """Base class for LCClass specializations for specific qubit numbers."""

    def __init__(self, type_or_id, data: linear_index.Repr | None = None):
        """Create an LC class from either a class id or an entanglument structure type
        together with further information on the exact LC class.

        Parameters
        ----------
        type_or_id : EntanglementStructure | int
            Either an entanglement structure key that is appropriate for the
            type of this class or an LC class id in form of an integer.
        data : Repr, optional
            Description of the qubit grouping, ignored if first parameter is an int.
        """
        cls = type(self)
        if isinstance(type_or_id, cls.EntanglementStructure):  # type: ignore
            self.type = type_or_id
            if data is None:
                data = linear_index.Repr()
            self.data = data
        elif isinstance(type_or_id, int):
            self._from_id(type_or_id)

    def __eq__(self, other) -> bool:
        return self.type == other.type and self.data == other.data

    def id(self) -> int:
        """Get a unique class id which is in the range [0, count()-1].

        Returns
        -------
        int
            LC equivalence class id
        """
        cls = type(self)
        com = cls.combinatorics[cls.combinatorics_map[self.type]]  # type: ignore
        return cls._start_indices[self.type] + com["to_lin_idx"](self.data)  # type: ignore

    def _from_id(self, id: int):
        """Assign this object to the LC equivalence class described by the given id"""
        cls = type(self)
        self.type = cls.get_entanglement_structure(id)
        class_index = id - cls._start_indices[self.type]  # type: ignore
        com = cls.combinatorics[cls.combinatorics_map[self.type]]  # type: ignore
        self.data = com["from_lin_idx1"](class_index)

    @abc.abstractmethod
    def num_qubits(self) -> int: pass

    @abc.abstractmethod
    def get_graph(self) -> Graph:
        """Get one representative graph that is local-Clifford equivalent to
        this LC class and thus all stabilizers that belong to it. Note that
        there are many equivalent graphs, connected by local complementation
        and the one returned here is sometimes somewhat arbitrary.

        Returns
        -------
        Graph
            Graph object
        """
        pass

    @classmethod
    def LC_GI_size(cls, structure) -> int:
        """Get the number of LC classes in one LC+GI class (one entanglement structure type)"""
        return cls._start_indices[structure+1] - cls._start_indices[structure]  # type: ignore

    @classmethod
    def get_LC_type(cls, id: int):
        """Generalized function to get the LC+GI equivalence class (Type) of a class id
        for a given type (LCClass2, LCClass3, LCClass4)

        Parameters
        ----------
        id : int
            LC equivalence class id
        cls : LCClass type, may be LCClass2, LCClass3, LCClass4, ...

        Returns
        -------
        cls.Type
            LC+GI equivalence class (Type) corresponding to the given id
        """
        assert 0 <= id < cls.count(), "Invalid id for equivalence classes LC(2)"
        for i, start_index in enumerate(cls._start_indices):  # type: ignore
            if start_index > id:
                return cls.EntanglementStructure(i - 1)  # type: ignore

    @classmethod
    def get_entanglement_structure(cls, id: int):
        """Get the entanglement structure type for given LC class id"""
        return cls.get_LC_type(id)

    @classmethod
    def count(cls) -> int:
        """Get the total number of 4-qubit LC classes"""
        return cls._start_indices[-1]  # type: ignore

    def __repr__(self) -> str:
        return f"Repr{self.id:self.data}"


class LCClass2(LCClassBase):

    @unique
    class EntanglementStructure(IntEnum):
        """2-Qubit LC+GI classes (entanglement structures)"""
        Separable = 0  # Separable graph
        Entangled = 1  # The vertices are connected by an edge

    _start_indices = [0, 1, 2]  # First index of each LC-GI-class / EntanglementStructure

    combinatorics = {
        "C0":  {"count": 1, "from_lin_idx1": linear_index.to_0, "to_lin_idx": linear_index.from_0},
    }

    combinatorics_map = {
        EntanglementStructure.Separable: "C0",
        EntanglementStructure.Entangled: "C0",
    }

    def num_qubits(self) -> int: return 2

    def get_graph(self) -> Graph:
        graph = Graph(2)
        if self.type == LCClass2.EntanglementStructure.Entangled:
            graph.add_edge(0, 1)
        return graph


class LCClass3(LCClassBase):

    @unique
    class EntanglementStructure(IntEnum):
        """3-Qubit LC+GI classes (entanglement structures)"""
        Separable = 0  # Separable graph
        Pair = 1       # One pair of connected vertices, the other vertices are isolated
        Triple = 2     # One triple of connected vertices, the remaining vertex is isolated

    _start_indices = [0, 1, 4, 5]  # First index of each LC-GI-class / EntanglementStructure

    combinatorics = {
        "C3":  {"count": 1, "from_lin_idx1": linear_index.to_0, "to_lin_idx": linear_index.from_0},
        "C12": {"count": 3, "from_lin_idx1": linear_index.to_12, "to_lin_idx": linear_index.from_12},
    }

    combinatorics_map = {
        EntanglementStructure.Separable: "C3",
        EntanglementStructure.Pair:      "C12",
        EntanglementStructure.Triple:    "C3",
    }

    def num_qubits(self) -> int: return 3

    def get_graph(self) -> Graph:
        graph = Graph(self.num_qubits())
        if self.type == LCClass3.EntanglementStructure.Pair:
            graph.add_edge(*self.data.get(2, 0))
        elif self.type == LCClass3.EntanglementStructure.Triple:
            graph.add_path([0, 1, 2])
        return graph


class LCClass4(LCClassBase):

    @unique
    class EntanglementStructure(IntEnum):
        """4-Qubit LC+GI classes (entanglement structures)"""
        Separable = 0  # Separable graph
        Pair = 1       # One pair of connected vertices, the other vertices are isolated
        Triple = 2     # One triple of connected vertices, the other vertices are isolated
        TwoPairs = 3   # Two pairs of connected vertices with no edges between them, the remaining vertex is isolated
        Star = 4       # A star of 4 vertices
        Line = 5       # A path graph of 4 vertices

    _start_indices = [0, 1, 7, 11, 14, 15, 18]  # First index of each LC-GI-class / EntanglementStructure

    combinatorics = {
        "C4":   {"count": 1, "from_lin_idx1": linear_index.to_0, "to_lin_idx": linear_index.from_0},
        "C112": {"count": 6, "from_lin_idx1": linear_index.to_112, "to_lin_idx": linear_index.from_112},
        "C13":  {"count": 4, "from_lin_idx1": linear_index.to_13, "to_lin_idx": linear_index.from_13},
        "C22":  {"count": 3, "from_lin_idx1": linear_index.to_22, "to_lin_idx": linear_index.from_22},
    }

    combinatorics_map = {
        EntanglementStructure.Separable: "C4",
        EntanglementStructure.Pair:      "C112",
        EntanglementStructure.Triple:    "C13",
        EntanglementStructure.TwoPairs:  "C22",
        EntanglementStructure.Star:      "C4",
        EntanglementStructure.Line:      "C22",
    }

    def num_qubits(self) -> int: return 4

    def get_graph(self) -> Graph:
        graph = Graph(self.num_qubits())
        if self.type == LCClass4.EntanglementStructure.Pair:
            graph.add_edge(*self.data.get(2, 0))
        elif self.type == LCClass4.EntanglementStructure.Triple:
            graph.add_path(self.data.get(3, 0))
        elif self.type == LCClass4.EntanglementStructure.TwoPairs:
            graph.add_edge(*self.data.get(2, 0))
            graph.add_edge(*self.data.get(2, 1))
        elif self.type == LCClass4.EntanglementStructure.Star:
            graph = Graph.star(self.num_qubits(), 0)
        elif self.type == LCClass4.EntanglementStructure.Line:
            graph.add_edge(*self.data.get(2, 0))
            graph.add_edge(*self.data.get(2, 1))
            graph.add_edge(self.data.get(2, 0)[0], self.data.get(2, 1)[1])
        return graph


class LCClass5(LCClassBase):

    @unique
    class EntanglementStructure(IntEnum):
        """5-Qubit LC+GI classes (entanglement structures)"""
        Separable = 0      # Separable graph
        Pair = 1           # One pair of connected vertices, the other vertices are isolated
        Triple = 2         # One triple of connected vertices, the other vertices are isolated
        TwoPairs = 3       # Two pairs of connected vertices with no edges between them, the remaining vertex is isolated
        Star4 = 4          # A star of 4 vertices
        Line4 = 5          # A path graph of 4 vertices
        Star = 6           # A star of 5 vertices
        PairAndTriple = 7  # One pair and one triple of connected vertices with no edges between them
        T = 8              # A star of four vertices where the remaining vertex is connected to one of the former
        Line = 9           # A path graph of 5 vertices
        Cycle = 10         # A cycle graph

    _start_indices = [0, 1, 11, 21, 36, 41, 56, 57, 67, 77, 92, 93]  # First index of each LC-GI-class / EntanglementStructure

    combinatorics = {
        "C5":   {"count": 1, "from_lin_idx1": linear_index.to_0, "to_lin_idx": linear_index.from_0},
        "C14":  {"count": 5, "from_lin_idx1": linear_index.to_14, "to_lin_idx": linear_index.from_14},
        "C122": {"count": 15, "from_lin_idx1": linear_index.to_122, "to_lin_idx": linear_index.from_122},
        "C23":  {"count": 10, "from_lin_idx1": linear_index.to_23, "to_lin_idx": linear_index.from_23},
    }

    combinatorics_map = {
        EntanglementStructure.Separable:     "C5",
        EntanglementStructure.Pair:          "C23",
        EntanglementStructure.Triple:        "C23",
        EntanglementStructure.TwoPairs:      "C122",
        EntanglementStructure.Star4:         "C14",
        EntanglementStructure.Line4:         "C122",
        EntanglementStructure.Star:          "C5",
        EntanglementStructure.PairAndTriple: "C23",
        EntanglementStructure.T:             "C23",
        EntanglementStructure.Line:          "C122",
        EntanglementStructure.Cycle:         "C5",
    }

    def num_qubits(self) -> int: return 5

    def get_graph(self) -> Graph:
        graph = Graph(self.num_qubits())
        if self.type == LCClass5.EntanglementStructure.Pair:
            graph.add_edge(*self.data.get(2, 0))
        elif self.type == LCClass5.EntanglementStructure.Triple:
            graph.add_path(self.data.get(3, 0))
        elif self.type == LCClass5.EntanglementStructure.TwoPairs:
            graph.add_edge(*self.data.get(2, 0))
            graph.add_edge(*self.data.get(2, 1))
        elif self.type == LCClass5.EntanglementStructure.Star4:
            graph = Graph.star(self.num_qubits(), self.data.get(4, 0)[0])
            graph.remove_all_edges_to(self.data.get(1, 0)[0])
        elif self.type == LCClass5.EntanglementStructure.Line4:
            graph.add_edge(*self.data.get(2, 0))
            graph.add_edge(*self.data.get(2, 1))
            graph.add_edge(self.data.get(2, 0)[0], self.data.get(2, 1)[0])
        elif self.type == LCClass5.EntanglementStructure.Star:
            graph = Graph.star(self.num_qubits(), 0)
        elif self.type == LCClass5.EntanglementStructure.PairAndTriple:
            graph.add_path(self.data.get(2, 0))
            graph.add_path(self.data.get(3, 0))
        elif self.type == LCClass5.EntanglementStructure.T:
            center = self.data.get(3, 0)[0]
            graph = Graph.star(self.num_qubits(), center)
            stem = self.data.get(2, 0)
            graph.remove_edge(stem[0], center)
            graph.add_edge(*stem)
        elif self.type == LCClass5.EntanglementStructure.Line:
            graph.add_path([*self.data.get(2, 0), *self.data.get(1, 0), *self.data.get(2, 1)])
        elif self.type == LCClass5.EntanglementStructure.Cycle:
            graph.add_path([0, 1, 2, 3, 4, 0])
        return graph


def determine_lc_class(stabilizer: Stabilizer) -> LCClass2 | LCClass3 | LCClass4 | LCClass5:
    """Determine the graph equivalence class under local complementation (LC class),
    that the stabilizer uniquely corresponds to. Every stabilizer state is local-Clifford
    equivalent to one or more graph states which are all in the same orbit under local
    complementation.

    Parameters
    ----------
    stabilizer : Stabilizer
        Stabilizer

    Returns
    -------
    LCClass2 | LCClass3 | LCClass4 | LCClass5
        LC class description together with type and vertex data.
    """
    num_qubits = stabilizer.num_qubits
    assert 2 <= num_qubits <= 5, "LC class determination is only supported for up to 5 qubits"
    if num_qubits == 2:
        return determine_lc_class2(stabilizer)
    elif num_qubits == 3:
        return determine_lc_class3(stabilizer)
    elif num_qubits == 4:
        return determine_lc_class4(stabilizer)
    elif num_qubits == 5:
        return determine_lc_class5(stabilizer)
    assert False


def determine_lc_class2(stabilizer: Stabilizer):
    is_qubit_entangled = [stabilizer.is_qubit_entangled(i) for i in range(stabilizer.num_qubits)]
    num_entangled_qubits = is_qubit_entangled.count(True)
    if num_entangled_qubits == 0:
        return LCClass2(LCClass2.EntanglementStructure.Separable)
    elif num_entangled_qubits == 2:
        return LCClass2(LCClass2.EntanglementStructure.Entangled)
    assert False, "Invalid stabilizer"


def determine_lc_class3(stabilizer: Stabilizer):
    is_qubit_entangled = [stabilizer.is_qubit_entangled(i) for i in range(stabilizer.num_qubits)]
    num_entangled_qubits = is_qubit_entangled.count(True)
    if num_entangled_qubits == 0:
        return LCClass3(LCClass3.EntanglementStructure.Separable)
    elif num_entangled_qubits == 3:
        return LCClass3(LCClass3.EntanglementStructure.Triple)
    elif num_entangled_qubits == 2:
        unentangled = is_qubit_entangled.index(False)
        entangled = [i for i, value in enumerate(is_qubit_entangled) if value]
        return LCClass3(LCClass3.EntanglementStructure.Pair, linear_index.Repr([[unentangled], entangled]))

    assert False, "Invalid stabilizer"


def determine_lc_class4(stabilizer: Stabilizer):
    is_qubit_entangled = [stabilizer.is_qubit_entangled(i) for i in range(stabilizer.num_qubits)]
    entangled_qubits = [i for i, value in enumerate(is_qubit_entangled) if value]
    unentangled_qubits = [i for i, value in enumerate(is_qubit_entangled) if not value]

    num_entangled_qubits = is_qubit_entangled.count(True)
    if num_entangled_qubits == 0:
        return LCClass4(LCClass4.EntanglementStructure.Separable)
    elif num_entangled_qubits == 2:
        return LCClass4(LCClass4.EntanglementStructure.Pair, linear_index.Repr([entangled_qubits, [unentangled_qubits[0]], [unentangled_qubits[1]]]))
    elif num_entangled_qubits == 3:
        return LCClass4(LCClass4.EntanglementStructure.Triple, linear_index.to_13(unentangled_qubits[0]))
    elif num_entangled_qubits == 4:
        # Can be TwoPairs, Line or Star
        X, Z = stabilizer.expand()
        identity_signature = (X.T | Z.T)

        def count_identity_string(identity_string):
            return len(np.where((identity_signature == identity_string).all(axis=1))[0])
        has_IIAA = count_identity_string([0, 0, 1, 1]) != 0
        has_IAIA = count_identity_string([0, 1, 0, 1]) != 0
        has_AIIA = count_identity_string([1, 0, 0, 1]) != 0

        if has_IIAA and has_AIIA:
            return LCClass4(LCClass4.EntanglementStructure.Star)

        A_n = count_identity_string([1, 1, 1, 1])  # last entry of the sector length distribution

        if has_IIAA:
            data = linear_index.to_22(0)
        elif has_IAIA:
            data = linear_index.to_22(1)
        elif has_AIIA:
            data = linear_index.to_22(2)
        else:
            assert False, "Invalid stabilizer"

        if A_n == 9:  # It is definitely TwoPairs
            return LCClass4(LCClass4.EntanglementStructure.TwoPairs, data)
        elif A_n == 5:
            return LCClass4(LCClass4.EntanglementStructure.Line, data)

    assert False, "Invalid stabilizer"


def count_identity_string(identity_signature, identity_string):
    return len(np.where((identity_signature == identity_string).all(axis=1))[0])


def determine_lc_class5(stabilizer: Stabilizer):
    num_qubits = stabilizer.num_qubits
    is_qubit_entangled = [stabilizer.is_qubit_entangled(i) for i in range(stabilizer.num_qubits)]
    entangled_qubits = [i for i, value in enumerate(is_qubit_entangled) if value]
    unentangled_qubits = [i for i, value in enumerate(is_qubit_entangled) if not value]

    num_entangled_qubits = is_qubit_entangled.count(True)
    if num_entangled_qubits == 0:
        return LCClass5(LCClass5.EntanglementStructure.Separable)
    elif num_entangled_qubits == 2:
        return LCClass5(LCClass5.EntanglementStructure.Pair, linear_index.Repr([entangled_qubits, unentangled_qubits]))
    elif num_entangled_qubits == 3:
        return LCClass5(LCClass5.EntanglementStructure.Triple, linear_index.Repr([entangled_qubits, unentangled_qubits]))

    else:
        X, Z = stabilizer.expand()
        identity_signature = (X.T | Z.T)

        def get_pairs(identity_signature, wanted_count) -> List[List[int]]:
            pairs = []
            for i, j in itertools.combinations(range(5), 2):
                signature = np.ones(num_qubits, dtype=np.int8)
                signature[i] = 0
                signature[j] = 0
                if count_identity_string(identity_signature, signature) == wanted_count:
                    pairs.append([i, j])
            return pairs

        def get_one_pair(identity_signature, wanted_count):
            pairs = get_pairs(identity_signature, wanted_count)
            assert len(pairs) == 1
            return pairs[0]

        def get_two_pairs(identity_signature, wanted_count):
            pairs = get_pairs(identity_signature, wanted_count)
            assert len(pairs) == 2
            return [pairs[0], pairs[1], [10 - sum(pairs[0]) - sum(pairs[1])]]

        def get_remaining_triple(pair):
            return list(filter(lambda el: not el in pair, range(5)))

        def identify_line5():
            return LCClass5(LCClass5.EntanglementStructure.Line, linear_index.Repr(get_two_pairs(identity_signature, 2)))

        def identify_line4():
            return LCClass5(LCClass5.EntanglementStructure.Line4, linear_index.Repr(get_two_pairs(identity_signature, 1)))

        def identify_two_pairs():
            return LCClass5(LCClass5.EntanglementStructure.TwoPairs, linear_index.Repr(get_two_pairs(identity_signature, 3)))

        def identify_star4():
            return LCClass5(LCClass5.EntanglementStructure.Star4, linear_index.Repr([[unentangled_qubits[0]], entangled_qubits]))

        def identify_pair_and_triple():
            pair = get_one_pair(identity_signature, 4)
            return LCClass5(LCClass5.EntanglementStructure.PairAndTriple, linear_index.Repr([pair, get_remaining_triple(pair)]))

        def identify_T():
            # A T or Pusteblume graph has a root and a stem. Each can be rotated around by local
            #    o                      graph complementations. The two stem vertices however always
            #    |     <- stem          stay the stem vertices. One of them is then connected to a
            #    o                      a third vertex which again is connected to the remaining two.
            #    |                      All graphs obtained by this construction are LC-equivalent as
            # o--o--o  <- root´         long as the stem vertices are the same. In the stabilizer the
            # operators with Pauli weight 4 and I at one of the stem indices occur 4 times. E.g. I····
            # and ·I···, while ··I··, ···I· and ····I occur only once.
            stem = []
            root = []
            for i in range(5):
                identity_string = np.ones(5)
                identity_string[i] = 0
                if count_identity_string(identity_signature, identity_string) == 4:
                    stem.append(i)
                else:
                    root.append(i)
            assert len(stem) == 2 and len(root) == 3, "Invalid stabilizer"
            return LCClass5(LCClass5.EntanglementStructure.T, linear_index.Repr([stem, root]))

        A_n = count_identity_string(identity_signature, [1, 1, 1, 1, 1])  # last entry of the sector length distribution

        if A_n == 9:
            # Construct any identity string with identity count 3 containing an entry at the unentangled qubit
            # The count of this operator is always 1 for a 4-star and 0 or 3 for two pairs.
            trial_identity_string = np.array(is_qubit_entangled)
            unentangled_qubit = unentangled_qubits[0]
            trial_identity_string[(unentangled_qubit + 1) % 5] = 0
            trial_identity_string[(unentangled_qubit + 2) % 5] = 0
            count = count_identity_string(identity_signature, trial_identity_string)
            if count == 1:
                return identify_star4()
            elif count == 3 or count == 0:
                return identify_two_pairs()
            else:
                assert False, "Invalid stabilizer"
        elif A_n == 5:
            return identify_line4()
        elif A_n == 16:
            return LCClass5(LCClass5.EntanglementStructure.Star)
        elif A_n == 12:
            return identify_pair_and_triple()
        elif A_n == 10:
            return identify_T()
        elif A_n == 8:
            return identify_line5()
        elif A_n == 6:
            return LCClass5(LCClass5.EntanglementStructure.Cycle)
    assert False, "Invalid stabilizer"
