from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


class RealAmplitude:
    def __init__(self, n_qubits, reps):
        self.circuit = QuantumCircuit(n_qubits)
        self.param = ParameterVector('θ', n_qubits * (reps + 1))
        for i in range(0, reps):
            for j in range(0, n_qubits):
                self.circuit.ry(self.param[i * n_qubits + j], j)
            for j in range(n_qubits - 1, 0, -1):
                self.circuit.cx(j - 1, j)
            self.circuit.barrier()

        for j in range(0, n_qubits):
            self.circuit.ry(self.param[reps * n_qubits + j], j)


class FakeAmplitude:
    def __init__(self, n_qubits, reps):
        # 创建量子线路
        self.circuit = QuantumCircuit(n_qubits)

        # 引入参数
        self.param = ParameterVector('θ', (n_qubits * (reps + 1)) * 2)

        for i in range(0, reps):
            for j in range(0, n_qubits):
                self.circuit.ry(self.param[i * n_qubits * 2 + 2 * j], j)
                self.circuit.rz(self.param[i * n_qubits * 2 + 2 * j + 1], j)
            for j in range(0, n_qubits):
                self.circuit.cz(j, (j + 1) % n_qubits)
            self.circuit.barrier()
        for j in range(0, n_qubits):
            self.circuit.ry(self.param[reps * n_qubits * 2 + 2 * j], j)
            self.circuit.rz(self.param[reps * n_qubits * 2 + 2 * j + 1], j)

