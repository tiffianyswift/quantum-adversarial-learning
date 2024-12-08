from copy import deepcopy

from qiskit.quantum_info import Operator

from qiskit.circuit.library import StatePreparation


from qiskit.primitives import Estimator
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

import numpy as np
import math

import os.path

from model_training_related.encoding import AmplitudeEncoding, AngleEncoding

np.set_printoptions(precision=65)  # 通常足够显示双精度浮点数的全部有效数字


# 幅度编码
class AmplitudeModel:
    def __init__(self, n_qubits, ansatz, observables):
        self.n_qubits = n_qubits
        self.ansatz = ansatz
        self.param = np.random.randn(len(self.ansatz.param))
        self.observables = observables.observable
        self.encoding = AmplitudeEncoding(n_qubits)
        self.circuit = self.encoding.circuit
        self.circuit.compose(self.ansatz.circuit, inplace=True)

    # 返回torch损失函数接受的loss向量格式
    def __call__(self, x):
        estimator = Estimator()
        # 将两个矩阵扁平化成一维数组
        num_samples = x.shape[0]
        x = x.view(num_samples, -1)
        params = np.broadcast_to(self.param, (num_samples, len(self.param)))

        theta = np.concatenate((x, params), axis=1)
        job = estimator.run([self.circuit] * num_samples, [self.observables] * num_samples, theta)

        return job.result().values

    def get_fixed_ansatz_circuit(self):
        return self.ansatz.get_fixed_ansatz_circuit(self.param)

    def dump(self, path):
        np.save(path, self.param)

    def load(self, path):
        if os.path.exists(path):
            self.param = np.load(path)

    # 求解析梯度
    # 参考论文：Yan J, Yan L, Zhang S. A new method of constructing adversarial examples for quantum variational circuits[J]. Chinese Physics B, 2023, 32(7): 070304.
    def solve_encoding_gradient(self, x):
        def get_grad_circuit(n_qubits, index):
            positions = [pos for pos, bit in enumerate(bin(index)[:1:-1], start=1) if bit == '1']
            grad_circuit = QuantumCircuit(n_qubits, name="get_amplitude_state_grad")
            for pos in positions:
                grad_circuit.cnot(0, pos)
            return grad_circuit

        def normalize_state(state_vector):
            eps = 1e-5
            while not math.isclose(sum(np.abs(state_vector) ** 2), 1.0, abs_tol=eps):
                state_vector /= np.linalg.norm(state_vector)
            return state_vector

        original_shape = x.shape
        num_samples = x.shape[0]
        x = x.view(num_samples, -1)
        grad = np.zeros(x.shape)
        for idx in range(num_samples):
            input_qc = QuantumCircuit(self.n_qubits + 1)
            target_state = x[idx].reshape(-1)
            target_state = normalize_state(target_state)

            target_state = np.array(target_state, dtype=np.float64)
            controlled_prepare = StatePreparation(target_state).control()

            input_qc.h(0)
            input_qc.append(controlled_prepare, range(self.n_qubits + 1))

            estimator = Estimator()

            for i, x_point in enumerate(target_state):
                print(idx, i)
                qc = deepcopy(input_qc)
                qc.barrier()
                grad_circuit = get_grad_circuit(self.n_qubits + 1, i)
                qc.x(0)
                qc.barrier()
                qc.compose(grad_circuit, inplace=True)
                qc.barrier()
                qc.x(0)
                qc.barrier()
                qc.compose(self.get_fixed_ansatz_circuit(), qubits=range(1, self.n_qubits + 1), inplace=True)
                qc.barrier()
                # 当matrix不是酉的，会存在问题
                matrix = self.observables.to_matrix()
                operator = Operator(matrix).to_instruction()
                qc.append(operator.control(), range(0, self.n_qubits + 1))

                qc.barrier()
                qc.h(0)
                qc.barrier()
                # 存在问题
                lis = [("I"*self.n_qubits+"Z", 2)]

                observable = SparsePauliOp.from_list(lis)
                job = estimator.run(qc, observable)

                grad[idx][i] = job.result().values[0]

        return grad.reshape(original_shape)

    # 求解估计梯度
    def evaluate_encoding_gradient(self, x, eps=1e-5):
        num_samples = x.shape[0]
        original_shape = x.shape
        x = x.view(num_samples, -1)
        original_x = deepcopy(x)
        grad = np.zeros(original_x.shape)
        for idx in range(num_samples):
            for i, x_point in enumerate(original_x[idx]):
                x[idx][i] = x_point + eps
                right = self(x[idx:idx+1])
                x[idx][i] = x_point - eps
                left = self(x[idx:idx+1])
                grad[idx][i] = (right-left) / (2*eps)
                x[idx][i] = x_point

        return grad

    def solve_ansatz_gradient(self, x):
        num_samples = x.shape[0]
        num_param = self.param.shape[0]
        original_param = self.param
        grad = np.zeros((num_samples, num_param))

        for i, param in enumerate(original_param):
            self.param[i] = param + np.pi / 2
            exps_right = self(x)
            self.param[i] = param - np.pi / 2
            exps_left = self(x)
            grad[:, i] = (exps_right - exps_left) / 2

            self.param[i] = param
        return grad

    def evaluate_ansatz_gradient(self, x, eps=1e-5):
        num_samples = x.shape[0]
        num_param = self.param.shape[0]
        original_param = self.param
        grad = np.zeros((num_samples, num_param))

        for i, param in enumerate(original_param):
            self.param[i] = param + eps
            right = self(x)
            self.param[i] = param - eps
            left = self(x)
            grad[:, i] = (right - left) / (2 * eps)

            self.param[i] = param
        return grad


class AngleModel:
    def __init__(self, n_qubits, n_features, ansatz, observables):
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.ansatz = ansatz
        self.param = np.random.randn(len(self.ansatz.param))
        self.observables = observables.observable
        self.encoding = AngleEncoding(self.n_qubits, self.n_features)
        self.circuit = self.encoding.circuit
        self.circuit.compose(self.ansatz.circuit, inplace=True)

    # 返回torch损失函数接受的loss向量格式
    def __call__(self, x):
        estimator = Estimator()
        # 将两个矩阵扁平化成一维数组
        num_samples = x.shape[0]
        x = x.view(num_samples, -1)
        params = np.broadcast_to(self.param, (num_samples, len(self.param)))

        theta = np.concatenate((x, params), axis=1)
        job = estimator.run([self.circuit] * num_samples, [self.observables] * num_samples, theta)

        return job.result().values

    def get_fixed_ansatz_circuit(self):
        return self.ansatz.get_fixed_ansatz_circuit(self.param)

    def dump(self, path):
        np.save(path, self.param)

    def load(self, path):
        if os.path.exists(path):
            self.param = np.load(path)

    # 求解析梯度
    def solve_encoding_gradient(self, x):
        num_samples = x.shape[0]
        original_shape = x.shape
        x = x.view(num_samples, -1)
        original_x = deepcopy(x)
        grad = np.zeros(original_x.shape)

        for idx in range(num_samples):
            for i, x_point in enumerate(original_x[idx]):
                x[idx][i] = x_point + np.pi / 2
                exps_right = self(x[idx:idx+1])
                x[idx][i] = x_point - np.pi / 2
                exps_left = self(x[idx:idx+1])
                grad[idx][i] = (exps_right - exps_left) / 2
                x[idx][i] = x_point

        return grad.reshape(original_shape)

    # 求解估计梯度
    def evaluate_encoding_gradient(self, x, eps=1e-5):
        num_samples = x.shape[0]
        original_shape = x.shape
        x = x.view(num_samples, -1)
        original_x = deepcopy(x)
        grad = np.zeros(original_x.shape)

        for idx in range(num_samples):
            for i, x_point in enumerate(original_x[idx]):
                x[idx][i] = x_point + eps
                right = self(x[idx:idx+1])
                x[idx][i] = x_point - eps
                left = self(x[idx:idx+1])
                grad[idx][i] = (right-left) / (2*eps)
                x[idx][i] = x_point
        return grad.reshape(original_shape)

    def solve_ansatz_gradient(self, x):
        num_samples = x.shape[0]
        num_param = self.param.shape[0]
        original_param = self.param
        grad = np.zeros((num_samples, num_param))

        for i, param in enumerate(original_param):
            self.param[i] = param + np.pi / 2
            exps_right = self(x)
            self.param[i] = param - np.pi / 2
            exps_left = self(x)
            grad[:, i] = (exps_right - exps_left) / 2

            self.param[i] = param
        return grad

    def evaluate_ansatz_gradient(self, x, eps=1e-5):
        num_samples = x.shape[0]
        num_param = self.param.shape[0]
        original_param = self.param
        grad = np.zeros((num_samples, num_param))

        for i, param in enumerate(original_param):
            self.param[i] = param + eps
            right = self(x)
            self.param[i] = param - eps
            left = self(x)
            grad[:, i] = (right - left) / (2 * eps)

            self.param[i] = param
        return grad















