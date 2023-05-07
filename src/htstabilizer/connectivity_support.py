

from typing import Literal


SupportedConnectivity = Literal["all", "linear", "star", "cycle", "T", "Q"]


def is_connectivity_supported(num_qubits: int, connectivity: SupportedConnectivity):
    if num_qubits < 2 or num_qubits > 5:
        return False
        raise ValueError(f"The given stabilizer has {num_qubits} qubits which is not supported.")

    return (num_qubits == 2 and connectivity == "all") or \
        (num_qubits == 3 and connectivity in ["all", "linear"]) or \
        (num_qubits == 4 and connectivity in ["all", "linear", "star", "cycle"]) or \
        (num_qubits == 5 and connectivity in ["all", "linear", "star", "cycle", "T",  "Q"])
