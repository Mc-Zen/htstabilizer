# Hardware-Tailored Stabilizer Circuits Python library

[![PyPI Package](https://img.shields.io/pypi/v/htstabilizer)](https://pypi.org/project/htstabilizer/)
[![MIT License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/Mc-Zen/htstabilizer/blob/master/LICENSE.txt)
[![Tests](https://github.com/Mc-Zen/htstabilizer/actions/workflows/run-tests.yml/badge.svg)](https://github.com/Mc-Zen/htstabilizer/actions/workflows/run-tests.yml)



This library provides 
- *hardware-tailored* quantum Clifford circuits for **stabilizer state preparation** or **measurement** as well as 
- mutually unbiased bases in order to perform more efficient **full-state tomography** on small qubit systems. 

Hereby, two-qubit gate count is minimized. All delivered circuits are designed to fully comply to a desired hardware connectivity, **avoiding additional swap operations altogether**. 


## Installation

You can install this package from [PyPi][pypi-page] via 
```
pip install htstabilizer
```
or you can clone the [repository][repository] and include it into your project. 

## Tailored to your hardware connectivity

A total of 19 connectivities are currently supported, ranging from 2 to 6 qubits. For each supported number of qubits, circuits for full connectivity and line connectivity are available. Additionally, other connectivities that occur in current quantum chips or which are subgraphs of existing hardware are supported. 

All currently available connectivities are shown below. 

![][hardware-connectivities]

<!-- ### 2 qubits
![][2-qubit-con]

### 3 qubits
![][3-qubit-con]

### 4 qubits
![][4-qubit-con]

### 5 qubits
![][5-qubit-con]

### 6 qubits
![][6-qubit-con] -->

## Usage

```py
from htstabilizer.stabilizer_circuits import *

pqc = get_preparation_circuit(Stabilizer(["XZZ", "ZXI", "ZIX"]), "linear")

rqc = get_readout_circuit(Stabilizer(["XZZ", "ZXI", "ZIX"]), "linear")

qc = QuantumCircuit(5)
# ... build Clifford circuit
compressed_qc = compress_preparation_circuit(qc, "T")
```


## Examples

View examples for exploring the functionality:

- [Compressing Clifford preparation circuits][example-compress]
- [Generating readout circuits][example-readout]
- [Perform state tomography][example-tomography]


## License
This package is distributed under the [MIT License][license]. 

[pypi-page]: https://pypi.org/project/htstabilizer/
[repository]: https://github.com/Mc-Zen/htstabilizer
[license]: https://github.com/Mc-Zen/htstabilizer/blob/master/LICENSE.txt

[hardware-connectivities]: https://github.com/Mc-Zen/htstabilizer/raw/master/docs/images/Hardware_Connectivities.svg

[example-compress]: https://github.com/Mc-Zen/htstabilizer/blob/master/examples/compress_preparation_circuit.py
[example-readout]: https://github.com/Mc-Zen/htstabilizer/blob/master/examples/readout_circuit.py
[example-tomography]: https://github.com/Mc-Zen/htstabilizer/blob/master/examples/state_tomography.py


