from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.circuit.library import RawFeatureVector


# 传入量子比特数，编码的维度为2**n_qubits
class AmplitudeEncoding:
    def __init__(self, n_qubits):
        self.n_qubits = int(n_qubits)
        self.circuit = RawFeatureVector(2 ** self.n_qubits)


# 传入量子比特数，和对应特征数量，编码维度为n_feature
class AngleEncoding:
    def __init__(self, n_qubits, n_features):
        # 创建量子线路
        self.circuit = QuantumCircuit(n_qubits)

        # 引入参数
        self.param = [Parameter(f'x[{i}]') for i in range(n_features)]

        for i in range(0, n_features):
            self.circuit.ry(self.param[i], i % n_qubits)

        self.circuit.barrier()
