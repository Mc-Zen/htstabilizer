""" 
This script computes and visualizes some statistics for the 
optimized stabilizer circuits. 

"""


from collections import defaultdict
from src.htstabilizer import stabilizer_circuits
from src.htstabilizer.lc_classes import *
from src.htstabilizer.circuit_lookup import stabilizer_circuit_lookup
import matplotlib.pyplot as plt


def stabilizer_circuit_statistics(num_qubits, connectivity):

    cost = defaultdict(int)
    depth = defaultdict(int)

    LCClasses = [LCClass2, LCClass3, LCClass4, LCClass5]
    cls = LCClasses[num_qubits - 2]

    for id in range(cls.count()):
        circuit_info = stabilizer_circuit_lookup(num_qubits, connectivity, id)
        cost[circuit_info.cost] += 1
        depth[circuit_info.depth] += 1

    average_cost = 0
    for the_cost, count in cost.items():
        average_cost += the_cost*count
    average_cost /= cls.count()

    average_depth = 0
    for the_depth, count in depth.items():
        average_depth += the_depth*count
    average_depth /= cls.count()

    max_cost = max(cost.keys())
    max_depth = max(depth.keys())

    print(f"Circuit statistics {connectivity}: \n avg 2-qubit count: {average_cost}\n"
          f" avg 2-qubit depth: {average_depth}\n max 2-qubit count: {max_cost}\n max 2-qubit depth: {max_depth}")

    plt.figure(figsize=(3, 4))
    plt.title(f"{num_qubits} qubits {connectivity}")

    # plt.plot(cost.keys(), cost.values(), label="cx count")
    plt.bar(list(cost.keys()), list(cost.values()), color=(1, 0, 0, .5), label="cx count")
    plt.bar(list(depth.keys()), list(depth.values()), color=(0, 1, 0, .5), label="cx depth")

    plt.legend()
    plt.tight_layout()


print("3 qubits")
stabilizer_circuit_statistics(3, "all")
stabilizer_circuit_statistics(3, "linear")

print("4 qubits")
stabilizer_circuit_statistics(4, "all")
stabilizer_circuit_statistics(4, "linear")
stabilizer_circuit_statistics(4, "star")
stabilizer_circuit_statistics(4, "cycle")

print("5 qubits")
stabilizer_circuit_statistics(5, "all")
stabilizer_circuit_statistics(5, "linear")
stabilizer_circuit_statistics(5, "star")
stabilizer_circuit_statistics(5, "cycle")
stabilizer_circuit_statistics(5, "T")
stabilizer_circuit_statistics(5, "Q")

plt.show()


# 4 Qubits: all > cycle, star > linear
# 5 Qubits: all > star > cycle > Q > T > linear
