from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


class RealAmplitude:
    def __init__(self, n_qubits, reps):
        self.n_qubits = n_qubits
        self.reps = reps
        self.circuit = QuantumCircuit(self.n_qubits)
        self.param = ParameterVector('θ', self.n_qubits * (self.reps + 1))

        for i in range(0, self.reps):
            for j in range(0, self.n_qubits):
                self.circuit.ry(self.param[i * self.n_qubits + j], j)
            # for j in range(self.n_qubits - 1, 0, -1):
            #     self.circuit.cx(j - 1, j)
            self.circuit.barrier()

        for j in range(0, self.n_qubits):
            self.circuit.ry(self.param[self.reps * self.n_qubits + j], j)

    def get_fixed_ansatz_circuit(self, param):
        fixed_circuit = QuantumCircuit(self.n_qubits)
        for i in range(0, self.reps):
            for j in range(0, self.n_qubits):
                fixed_circuit.ry(param[i * self.n_qubits + j], j)
            # for j in range(self.n_qubits - 1, 0, -1):
            #     fixed_circuit.cx(j-1, j)
            fixed_circuit.barrier()

        for j in range(0, self.n_qubits):
            fixed_circuit.ry(param[self.reps * self.n_qubits + j], j)

        return fixed_circuit


class FakeAmplitude:
    def __init__(self, n_qubits, reps):
        self.n_qubits = n_qubits
        self.reps = reps
        self.circuit = QuantumCircuit(self.n_qubits)
        self.param = ParameterVector('θ', (self.n_qubits * (self.reps + 1)) * 2)

        for i in range(0, self.reps):
            for j in range(0, self.n_qubits):
                self.circuit.ry(self.param[i * self.n_qubits * 2 + 2 * j], j)
                self.circuit.rz(self.param[i * self.n_qubits * 2 + 2 * j + 1], j)
            for j in range(0, self.n_qubits):
                self.circuit.cz(j, (j + 1) % self.n_qubits)
            self.circuit.barrier()
        for j in range(0, self.n_qubits):
            self.circuit.ry(self.param[self.reps * self.n_qubits * 2 + 2 * j], j)
            self.circuit.rz(self.param[self.reps * self.n_qubits * 2 + 2 * j + 1], j)

    def get_fixed_ansatz_circuit(self, param):
        fixed_circuit = QuantumCircuit(self.n_qubits)
        for i in range(0, self.reps):
            for j in range(0, self.n_qubits):
                fixed_circuit.ry(param[i * self.n_qubits * 2 + 2 * j], j)
                fixed_circuit.ry(param[i * self.n_qubits * 2 + 2 * j + 1], j)
            for j in range(0, self.n_qubits):
                fixed_circuit.cz(j, (j + 1) % self.n_qubits)
            fixed_circuit.barrier()

        for j in range(0, self.n_qubits):
            fixed_circuit.ry(param[self.reps * self.n_qubits * 2 + 2 * j], j)
            fixed_circuit.rz(param[self.reps * self.n_qubits * 2 + 2 * j + 1], j)

        return fixed_circuit

