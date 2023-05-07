
from src.htstabilizer.stabilizer_circuits import *
from src.htstabilizer.mub_circuits import *
import qiskit
from qiskit import QuantumRegister
from src.htstabilizer.tomography import *

pqc = get_preparation_circuit(Stabilizer(["XZZ", "ZXI", "ZIX"]), "linear")

rqc = get_readout_circuit(Stabilizer(["XZZ", "ZXI", "ZIX"]), "linear")


q2 = QuantumRegister(6)
bell = QuantumCircuit(q2)
bell.h(3)
bell.cx(3, 5)
print(bell)
qst = stabilizer_measurement_circuit(bell, Stabilizer(["XI", "IZ"]), "all", [3,5]) 
print(qst)

# bell = QuantumCircuit(2)
# bell.h(0)
# bell.cx(0, 1)
# print(bell)
# qst = stabilizer_measurement_circuit(bell, Stabilizer(["XI", "IZ"]), "all")
# print(qst)

backend_sim = qiskit.Aer.get_backend("qasm_simulator")
job_sim = qiskit.execute(qst, backend=backend_sim, shots=100)
result = job_sim.result()

fitter = StabilizerMeasurementFitter(result, qst)
print(fitter.expectation_values())
