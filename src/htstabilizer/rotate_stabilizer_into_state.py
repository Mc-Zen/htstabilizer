from typing import Union, Collection
from qiskit import QuantumCircuit
from qiskit.quantum_info import StabilizerState, Pauli, Statevector
import numpy as np

from . import f2_algebra as f2
from .stabilizer import Stabilizer


def synth_circuit_from_stabilizers(
    stabilizers: Collection[str],
    allow_redundant: bool = False,
    allow_underconstrained: bool = False,
    invert: bool = False,
) -> QuantumCircuit:
    "Taken from qiskit (this function is only available from qiskit-1.0.0 on)"
    # pylint: disable=line-too-long
    """Synthesis of a circuit that generates a state stabilized by the stabilizers
    using Gaussian elimination with Clifford gates.
    If the stabilizers are underconstrained, and ``allow_underconstrained`` is ``True``,
    the circuit will output one of the states stabilized by the stabilizers.
    Based on stim implementation.

    Args:
        stabilizers: List of stabilizer strings
        allow_redundant: Allow redundant stabilizers (i.e., some stabilizers
            can be products of the others)
        allow_underconstrained: Allow underconstrained set of stabilizers (i.e.,
            the stabilizers do not specify a unique state)
        invert: Return inverse circuit

    Returns:
        A circuit that generates a state stabilized by ``stabilizers``.

    Raises:
        ValueError: if the stabilizers are invalid, do not commute, or contradict each other,
                     if the list is underconstrained and ``allow_underconstrained`` is ``False``,
                     or if the list is redundant and ``allow_redundant`` is ``False``.

    References:
        1. https://github.com/quantumlib/Stim/blob/c0dd0b1c8125b2096cd54b6f72884a459e47fe3e/src/stim/stabilizers/conversions.inl#L469
        2. https://quantumcomputing.stackexchange.com/questions/12721/how-to-calculate-destabilizer-group-of-toric-and-other-codes

    """
    from qiskit.quantum_info import PauliList, Clifford
    stabilizer_list = PauliList(stabilizers)
    if np.any(stabilizer_list.phase % 2):
        raise ValueError("Some stabilizers have an invalid phase")
    if len(stabilizer_list.commutes_with_all(stabilizer_list)) < len(stabilizer_list):
        raise ValueError("Some stabilizers do not commute.")

    num_qubits = stabilizer_list.num_qubits
    circuit = QuantumCircuit(num_qubits)

    used = 0
    for i in range(len(stabilizer_list)):
        curr_stab = stabilizer_list[i].evolve(Clifford(circuit), frame="s")

        # Find pivot.
        pivot = used
        while pivot < num_qubits:
            if curr_stab[pivot].x or curr_stab[pivot].z:
                break
            pivot += 1

        if pivot == num_qubits:
            if curr_stab.x.any():
                raise ValueError(
                    f"Stabilizer {i} ({stabilizer_list[i]}) anti-commutes with some of "
                    "the previous stabilizers."
                )
            if curr_stab.phase == 2:
                raise ValueError(
                    f"Stabilizer {i} ({stabilizer_list[i]}) contradicts "
                    "some of the previous stabilizers."
                )
            if curr_stab.z.any() and not allow_redundant:
                raise ValueError(
                    f"Stabilizer {i} ({stabilizer_list[i]}) is a product of the others "
                    "and allow_redundant is False. Add allow_redundant=True "
                    "to the function call if you want to allow redundant stabilizers."
                )
            continue

        # Change pivot basis to the Z axis.
        if curr_stab[pivot].x:
            if curr_stab[pivot].z:
                circuit.h(pivot)
                circuit.s(pivot)
                circuit.h(pivot)
                circuit.s(pivot)
                circuit.s(pivot)
            else:
                circuit.h(pivot)

        # Cancel other terms in Pauli string.
        for j in range(num_qubits):
            if j == pivot or not (curr_stab[j].x or curr_stab[j].z):
                continue
            p = curr_stab[j].x + curr_stab[j].z * 2
            if p == 1:  # X
                circuit.h(pivot)
                circuit.cx(pivot, j)
                circuit.h(pivot)
            elif p == 2:  # Z
                circuit.cx(j, pivot)
            elif p == 3:  # Y
                circuit.h(pivot)
                circuit.s(j)
                circuit.s(j)
                circuit.s(j)
                circuit.cx(pivot, j)
                circuit.h(pivot)
                circuit.s(j)

        # Move pivot to diagonal.
        if pivot != used:
            circuit.swap(pivot, used)

        # fix sign
        curr_stab = stabilizer_list[i].evolve(Clifford(circuit), frame="s")
        if curr_stab.phase == 2:
            circuit.x(used)
        used += 1

    if used < num_qubits and not allow_underconstrained:
        raise ValueError(
            "Stabilizers are underconstrained and allow_underconstrained is False."
            " Add allow_underconstrained=True  to the function call "
            "if you want to allow underconstrained stabilizers."
        )
    if invert:
        return circuit
    return circuit.inverse()


def rotate_stabilizer_into_state(
        circuit: QuantumCircuit,
        target: Union[QuantumCircuit, Stabilizer],
        inplace=False
) -> QuantumCircuit:
    """
    Given a stabilizer preparation circuit and a stabilizer that represent 
    states with the same stabilizer group up to phases, modify the circuit,
    so that it has the same stabilizer as the target stabilizer.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to modify so that it prepares the same state as `target`
    target : QuantumCircuit | Stabilizer
        Target stabilizer or preparation circuit. 
    inplace : bool, optional
        Whether to perform the modification inplace or return a new circuit.
        If True, the input circuit is modified and returned, by default False

    Returns
    -------
    QuantumCircuit
        Modified `circuit`
    """
    if isinstance(target, Stabilizer):
        # from qiskit.synthesis import synth_circuit_from_stabilizers
        target = synth_circuit_from_stabilizers(target.to_list(qiskit_convention=True))
        return _rotate_stabilizer_into_state_circuit(circuit, target, inplace=inplace)
        # from qiskit.quantum_info import StabilizerState
        # from qiskit import transpile
        # qiskit_stabilizer = StabilizerState.from_stabilizer_list(target.to_list())
        # qc = synth_stabilizer_layers(qiskit_stabilizer)
        # target = transpile(qc, basis_gates=["h", "x", "sx", "s", "sdg", "cz", "y", "z"])

    if isinstance(target, QuantumCircuit):
        return _rotate_stabilizer_into_state_circuit(circuit, target, inplace=inplace)

    if isinstance(target, Stabilizer):
        return _rotate_stabilizer_into_state_stabilizer(circuit, target, inplace=inplace)


def _rotate_stabilizer_into_state_circuit(
        circuit: QuantumCircuit,
        target: QuantumCircuit,
        inplace=False
) -> QuantumCircuit:
    """
    Given two stabilizer preparation circuits that prepare states 
    with the same stabilizer group up to phases, modify the first
    circuit, so that it prepares the same state as the target circuit.
    They then also prepare the exact same stabilizer group (but the 
    canonical generator may of course still not be the same). 

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to modify so that it prepares the same state as `target`
    target : QuantumCircuit
        Circuit that prepares the target state
    inplace : bool, optional
        Whether to perform the modification inplace or return a new circuit.
        If True, the input circuit is modified and returned, by default False

    Returns
    -------
    QuantumCircuit
        Modified `circuit`
    """

    # The generators G1, G2 of the stabilizers S1, S2 of the circuits U1, U2
    # are generally not the same, so comparing the phase does not work.
    # Instead we take the stabilizer of the first circuit, and evolve the Paulis
    # back through U2^\dagger. This should result in Paulis of the form +-Z^s,
    # the sign being negative whenever the Pauli had the wrong phase.
    # We then prepend an X gate to the circuit on the corresponding qubit.
    stabilizer = [Pauli(label) for label in StabilizerState(
        circuit).clifford.to_labels(mode="S")]

    target_inv = target.inverse()
    pauli_layer = QuantumCircuit(circuit.num_qubits)
    found_phase_mismatch = False

    for index, pauli in enumerate(stabilizer):
        p = pauli.evolve(target_inv, frame="s")
        if p.x.any():  # check that p has no x components
            raise ValueError("The two given circuits do not prepare the same stabilizer")
        if p.phase == 2:
            pauli_layer.x(index)
            found_phase_mismatch = True

    if not found_phase_mismatch:
        return circuit

    result = circuit.compose(pauli_layer, front=True, inplace=inplace)
    # if inplace is False, compose() returns None
    return circuit if result is None else result

    """
    XX,YY
    1111
    0011

    -XX,ZZ   => mit Circuit
    1100
    0011


    """


def _rotate_stabilizer_into_state_stabilizer(
        circuit: QuantumCircuit,
        target: Stabilizer,
        inplace=False
) -> QuantumCircuit:
    raise NotImplementedError(
        "This function does not work correctly. Don't use it right now")

    """
    Given a stabilizer preparation circuit and a stabilizer that represent 
    states with the same stabilizer group up to phases, modify the circuit,
    so that it has the same stabilizer as the target stabilizer.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to modify so that it prepares the same state as `target`
    target : Stabilizer
        target stabilizer
    inplace : bool, optional
        Whether to perform the modification inplace or return a new circuit.
        If True, the input circuit is modified and returned, by default False

    Returns
    -------
    QuantumCircuit
        Modified `circuit`
    """

    # How does it work:
    #
    # Let G=[R S], G'=[R' S'] and q, q' phase vectors
    # Let M, M' ∈ GL_n(F2) basis change matrices so that
    #     M[G, q] = [Grref, qr]
    #     M'[G', q'] = [Grref', qr']
    #
    # Then Grref=Grref' and qr=qr' iff cal(S)=cal(S)', cal(S)=<(-1)^qi i^(sum_j r_ij s_ij) X^ri Z^si | i=1,...,n>
    #
    # In that case, we have q'=(M')^-1 qr' = (M')^-1 qr = (M')^-1 M q
    #
    # Now: G' is the generator that we have, and G is where we want to go
    # (note: we need <G> = <G'> of course). We therefore take q and compute
    # q' from the calculation above. We then compare q' with the phases p' of
    # the stabilizer (spanned by G) that we currently have and insert X gates
    # at the front of the circuit where the phases q' and p' differ.

    n = circuit.num_qubits
    src = Stabilizer(circuit)
    G = np.hstack((target.R.T, target.S.T, target.phases.reshape((n, 1))))
    q = target.phases
    G_prime = np.hstack((src.R.T, src.S.T, src.phases.reshape((n, 1))))
    p_prime = src.phases

    rref, M, M_inv = f2.rref_and_basis_change(G)
    rref_prime, M_prime, M_prime_inv = f2.rref_and_basis_change(G_prime)
    np.testing.assert_array_equal(f2.mat_mul(M, G), rref)
    np.testing.assert_array_equal(f2.mat_mul(M_prime, G_prime), rref_prime)
    np.testing.assert_array_equal(f2.mat_mul(M, M_inv), np.eye(M.shape[0]))
    np.testing.assert_array_equal(f2.mat_mul(M_prime, M_prime_inv), np.eye(M.shape[0]))
    np.testing.assert_array_equal(f2.mat_mul(M_prime, p_prime), rref_prime[:, 2*n])
    # np.testing.assert_array_equal(rref_prime, rref)
    gdiff = f2.add(rref[:, 2*n], rref_prime[:, 2*n])
    # gdiff = f2.mat_mul(M_prime_inv, gdiff)

    q_prime = f2.mat_mul(f2.mat_mul(M_prime_inv, M), q)
    # q_prime = f2.mat_mul(M_prime_inv, G[:,2*n])
    # q_prime = G[:,2*n]
    # print(G[:,2*n], G_prime[:,2*n], gdiff, f2.add(q_prime, p_prime))

    pauli_layer = QuantumCircuit(circuit.num_qubits)
    found_phase_mismatch = False
    for qubit in range(src.num_qubits):
        # if q_prime[qubit] != p_prime[qubit]:
        if gdiff[qubit]:
            pauli_layer.x(qubit)
            found_phase_mismatch = True

    if not found_phase_mismatch:
        return circuit

    result = circuit.compose(pauli_layer, front=True, inplace=inplace)
    # np.testing.assert_array_equal(Stabilizer(result).phases, q_prime)
    # np.testing.assert_array_equal(f2.mat_mul(M_prime, Stabilizer(result).phases), rref[:,2*n])

    # if inplace is False, compose() returns None
    return circuit if result is None else result


def assert_same_state(circuit1: QuantumCircuit, circuit2: QuantumCircuit):
    """
    Assert that two circuits prepare the same state when starting 
    from |0〉 (up to a global phase). 
    """
    statevector1 = np.array(Statevector(circuit1))
    statevector2 = np.array(Statevector(circuit2))
    import qiskit.quantum_info as qi
    qi.state_fidelity(statevector1, statevector2)
    return np.testing.assert_almost_equal(qi.state_fidelity(statevector1, statevector2), 1.0)
    max_index = np.argmax(np.abs(statevector1))
    if isinstance(max_index, np.ndarray):
        max_index = max_index[0]
    ratio = statevector1[max_index] / statevector2[max_index]
    np.testing.assert_array_almost_equal(statevector1, statevector2 * ratio)


def do_prepare_same_state(circuit1: QuantumCircuit, circuit2: QuantumCircuit) -> bool:
    """
    Check if two circuits prepare the same state when starting 
    from |0〉 (up to a global phase). 
    """
    try:
        assert_same_state(circuit1, circuit2)
        return True
    except:
        return False
