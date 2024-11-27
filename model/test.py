from copy import deepcopy

import math

import torch
import numpy as np

from ansatz import RealAmplitude
from model import AngleModel, AmplitudeModel
from observable import TwoClassifyObservable
from utils.util import get_grad
from qiskit.circuit.library import StatePreparation

from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import Pauli
from qiskit import quantum_info


def get_adv_grad(param_circuit, inputs, observable):
    def get_grad_circuit(n_qubits, index):
        positions = [pos for pos, bit in enumerate(bin(index)[:1:-1], start=1) if bit == '1']
        grad_circuit = QuantumCircuit(n_qubits, name="get_grad")
        for pos in positions:
            grad_circuit.cnot(0, pos)
        return grad_circuit

    def normalize_state(state_vector):
        eps = 1e-10
        while not math.isclose(sum(np.abs(state_vector) ** 2), 1.0, abs_tol=eps):
            state_vector /= np.linalg.norm(state_vector)
        return state_vector

    n_qubits = param_circuit.num_qubits
    input_qc = QuantumCircuit(n_qubits + 1)
    target_state = normalize_state(inputs)

    controlled_prepare = StatePreparation(target_state).control()

    input_qc.h(0)
    input_qc.append(controlled_prepare, range(n_qubits + 1))

    grad_list = []

    for index, _ in enumerate(inputs):
        qc = deepcopy(input_qc)
        grad_circuit = get_grad_circuit(n_qubits + 1, index)
        qc.x(0)
        qc.compose(grad_circuit, inplace=True)
        qc.x(0)
        qc.compose(param_circuit, inplace=True)
        qc.cz(control_qubit=0, target_qubit=1)
        qc.h(0)

        qc.save_expectation_value(quantum_info.Pauli(observable), [0], observable)

        transpiled_qc = transpile(qc, Aer.get_backend('qasm_simulator'))
        job = Aer.get_backend('qasm_simulator').run(transpiled_qc)
        grad_list.append(job.result().data()[observable])

    return grad_list




# 获取对抗样本需要添加的梯度
# param_circuit 被攻击的模型的参数化量子线路
# inputs 用于计算梯度的样本
# observable param_circuit所使用的可观测量

def getAdvGrad(param_circuit, inputs, observable):
    # 获取某个态对于某个坐标的倒数，其结果也是一个态
    def getGrad(n_qubits, num):
        positions = []
        position = 1

        while num > 0:
            if num & 1:
                positions.append(position)
            num >>= 1
            position += 1

        qc = QuantumCircuit(n_qubits, name="getGrad")
        for item in positions:
            qc.cnot(0, item)
        return qc
    n_qubits = param_circuit.num_qubits
    inputqc = QuantumCircuit(n_qubits + 1)

    target_state = inputs / np.linalg.norm(inputs)
    target_state = np.array(target_state, dtype=np.float64)
    _EPS = 1e-10
    while not math.isclose(sum(np.absolute(target_state) ** 2), 1.0, abs_tol=_EPS):
        norm = np.sqrt(sum(np.abs(target_state) ** 2))
        target_state = target_state / norm

    controlled_prepare = StatePreparation(target_state).control()

    inputqc.h(0)
    inputqc.append(controlled_prepare, range(n_qubits + 1))

    grad_list = []

    for index, datapoint in enumerate(inputs):
        print(index)
        qc = deepcopy(inputqc)
        grad_circuit = getGrad(n_qubits + 1, index)
        qc.x(0)
        qc.append(grad_circuit, range(n_qubits + 1))
        qc.x(0)

        # qc.append(param_circuit, range(1, n_qubits + 1))
        qc.cz(control_qubit=0, target_qubit=1)
        qc.h(0)

        qc.save_expectation_value(Pauli(observable), [0], observable)

        transpiled_qc = transpile(qc, Aer.get_backend('qasm_simulator'))
        job = Aer.get_backend('qasm_simulator').run(transpiled_qc)
        grad_list.append(job.result().data()[observable])
    return grad_list

if __name__ == '__main__':
    ansat = RealAmplitude(4, 2)
    observable = TwoClassifyObservable(4)
    model1 = AmplitudeModel(4, ansat, observable)
    loss_func = torch.nn.BCELoss()
    data = torch.tensor(np.random.randn(1, 16))
    target = torch.tensor([1])
    output = torch.tensor(model1(data), dtype=torch.float32, requires_grad=True)
    loss = loss_func(output, target.float())
    loss.backward()
    grad1 = torch.tensor(get_grad(model1, data, np.array(output.grad)), dtype=torch.float32)
    grad2 = model1.solve_ansatz_gradient(data, np.array(output.grad))

    qc = getAdvGrad(model1.ansatz.circuit, data.numpy()[0], "z")
    print(qc)

# data = torch.tensor()
    #
    # print(model1(data))

