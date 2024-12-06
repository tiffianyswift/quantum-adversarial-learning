from qiskit.quantum_info import SparsePauliOp


class TwoClassifyObservable:
    def __init__(self, n_qubits):
        # lis = [("I"*(n_qubits-1)+"Z", 1 / 2), ("I"*n_qubits, 1 / 2)]
        lis = [("I" * (n_qubits - 1) + "Z", 1)]
        self.observable = SparsePauliOp.from_list(lis)
