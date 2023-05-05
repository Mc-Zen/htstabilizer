# Hardware-Tailored Stabilizer Circuits Python library

This library provides 
- *hardware-tailored* quantum Clifford circuits for **stabilizer state preparation** or **measurement** as well as 
- mutually unbiased bases in order to perform efficient **full-state tomography** on small qubit systems. 

Hereby, two-qubit gate count is minimized. All delivered circuits are designed to fully comply to a desired hardware connectivity, **avoiding additional swap operations altogether**. 


## Installation

The library can be easily installed from [PyPi][pypi-page]:
```
pip install htstabilizer
```

## Tailored to your hardware connectivity

A total of 13 connectivities are currently supported, ranging from 2 to 5 qubits. For each supported number of qubits, circuits for full connectivity and line connectivity are available. Additionally, other connectivities that occur in current quantum chips or which are subgraphs of existing hardware are supported. 

The following graphics show all available connectivities. 

### 2 qubits
![][2-qubit-con]

### 3 qubits
![][3-qubit-con]

### 4 qubits
![][4-qubit-con]

### 5 qubits
![][5-qubit-con]

## Usage

```
from htstabilizer.stabilizer_circuits import *

pqc = get_preparation_circuit(Stabilizer(["XZZ", "ZXI", "ZIX"]), "linear")

rqc = get_readout_circuit(Stabilizer(["XZZ", "ZXI", "ZIX"]), "linear")

```


[pypi-page]: https://pypi.org/project/shortcircuits/
[2-qubit-con]: docs/images/2-qubit%20connectivities.png
[3-qubit-con]: docs/images/3-qubit%20connectivities.png
[4-qubit-con]: docs/images/4-qubit%20connectivities.png
[5-qubit-con]: docs/images/5-qubit%20connectivities.png
