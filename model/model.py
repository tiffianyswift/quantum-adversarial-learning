from ansatz import RealAmplitude, FakeAmplitude
from encoding import AmplitudeEncoding, AngleEncoding
from observable import TwoClassifyObservable

from qiskit.primitives import Estimator

import numpy as np
import torch

import os.path

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

    def dump(self, path):
        np.save(path, self.param)

    def load(self, path):
        if os.path.exists(path):
            self.param = np.load(path)

    # 求解析梯度
    def solve_encoding_gradient(self):
        pass

    # 求解估计梯度
    def evaluate_encoding_gradient(self):
        pass

    def solve_ansatz_gradient(self):
        pass

    def evaluate_ansatz_gradient(self):
        pass


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

    def dump(self, path):
        np.save(path, self.param)

    def load(self, path):
        if os.path.exists(path):
            self.param = np.load(path)

    # 求解析梯度
    def solve_encoding_gradient(self):
        pass

    # 求解估计梯度
    def evaluate_encoding_gradient(self):
        pass

    def solve_ansatz_gradient(self):
        pass

    def evaluate_ansatz_gradient(self):
        pass


if __name__ == '__main__':
    ansat = RealAmplitude(4, 2)
    observable = TwoClassifyObservable(4)
    model1 = AngleModel(4, 5, ansat, observable)
    x = torch.tensor(np.array([[1, 2, 3, 4, 5]]))
    output = model1(x)
    print(output)










