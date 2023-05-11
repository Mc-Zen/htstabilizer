
from htstabilizer.tomography import *
import qiskit


backend = qiskit.Aer.get_backend("qasm_simulator")


# Create a preparation circuit for the state to measure
preparation_circuit = QuantumCircuit(5)
preparation_circuit.h(0)
preparation_circuit.cx(range(4), range(1, 5))


##############################
# Full state tomography
##############################

# Get 2^5 optimized circuits for measurement on linear hardware connectivity
qst = full_state_tomography_circuits(preparation_circuit, "linear")
job = qiskit.execute(qst, backend=backend, shots=100)

fitter = FullStateTomographyFitter(job.result(), qst)

# We can get a dictionaray of expectation values
print(fitter.expectation_values())
# Or the state matrix
print(fitter.density_matrix())


# It is possible to measure marginals by passing the a list of qubits to measure
qst = full_state_tomography_circuits(preparation_circuit, "all", measured_qubits=[0, 1])
job = qiskit.execute(qst, backend=backend, shots=100)

fitter = FullStateTomographyFitter(job.result(), qst)
print(fitter.density_matrix())




##############################
# Measure only a stabilizer group
##############################

smc = stabilizer_measurement_circuit(
    preparation_circuit,
    Stabilizer(["XX", "YY"]),
    "all",
    measured_qubits=[0, 1]
)
job = qiskit.execute(smc, backend=backend, shots=100)

fitter = StabilizerMeasurementFitter(job.result(), smc)
print(fitter.expectation_values())
