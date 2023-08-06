from typing import Union
from qiskit import QuantumCircuit
from qiskit.quantum_info import StabilizerState, Pauli, Statevector
import numpy as np

from . import f2_algebra as f2
from .stabilizer import Stabilizer


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
    stabilizer = [Pauli(label) for label in StabilizerState(circuit).clifford.to_labels(mode="S")]

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
    return circuit if result is None else result  # if inplace is False, compose() returns None

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
    # Let G=[R S], G'=[R' S'] and q,q' phase vectors
    # Let M, M' ∈ GL_n(F2) basis change matrices so that
    #     M[G, q] = [Grref, qr]
    #     M'[G', q'] = [Grref', qr']
    #
    # Then Grref=Grref' and qr=qr' iff γ=γ', γ=<(-1)^qi i^(sum_j r_ij s_ij) X^ri Z^si | i=1,...,n>
    #
    # In that case, we have q'=(M')^-1 qr' = (M')^-1 qr = (M')^-1 M q
    #
    # Now: G' is the generator that we have, and G is where we want to go
    # (note: we need <G> = <G'> of course). We therefore take q and compute
    # q' from the calculation above. We then compare q' with the phases p' of
    # the stabilizer (spanned by G) that we currently have and insert X gates
    # at the front of the circuit where the phases q' and p' differ.

    src = Stabilizer(circuit)
    G = np.hstack((target.R.T, target.S.T))
    q = target.phases
    G_prime = np.hstack((src.R.T, src.S.T))
    p_prime = src.phases

    rref, M, M_inv = f2.rref_and_basis_change(G)
    rref_prime, M_prime, M_prime_inv = f2.rref_and_basis_change(G_prime)
    # np.testing.assert_array_equal(rref_prime, rref)

    q_prime = f2.mat_mul(f2.mat_mul(M_prime_inv, M), q)

    pauli_layer = QuantumCircuit(circuit.num_qubits)
    found_phase_mismatch = False
    for qubit in range(src.num_qubits):
        if q_prime[qubit] != p_prime[qubit]:
            pauli_layer.x(qubit)
            found_phase_mismatch = True

    if not found_phase_mismatch:
        return circuit

    result = circuit.compose(pauli_layer, front=True, inplace=inplace)
    # np.testing.assert_array_equal(Stabilizer(result).phases, q_prime)

    return circuit if result is None else result  # if inplace is False, compose() returns None


def assert_same_state(circuit1: QuantumCircuit, circuit2: QuantumCircuit):
    """
    Assert that two circuits prepare the same state when starting 
    from |0〉 (up to a global phase). 
    """
    statevector1 = np.array(Statevector(circuit1))
    statevector2 = np.array(Statevector(circuit2))
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
