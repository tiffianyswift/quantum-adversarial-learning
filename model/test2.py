import qiskit
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit.library import StatePreparation
from qiskit_machine_learning.circuit.library import RawFeatureVector
import torch
from copy import deepcopy
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import math
from qiskit import quantum_info

from ansatz import RealAmplitude
from model import AmplitudeModel
from observable import TwoClassifyObservable
from qiskit.primitives import Estimator

n_qubits = 8
q_depth = 2





class Hybrid(torch.nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, q_depth, backend)
        self.q_params = torch.nn.Parameter(q_delta * torch.randn((q_depth+1) * n_qubits*3))
    def forward(self, inputv):
        return HybridFunction.apply(inputv, self.q_params, self.quantum_circuit)

# getUtheta 获取分类器的参数化量子线路
# path 模型的路径
# 获取模型参数，构造分类器线路
def getUtheta():
    model = torch.load("..\model_saved\model_epoch_13.pth")


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


# 获取对抗样本需要添加的梯度
# param_circuit 被攻击的模型的参数化量子线路
# inputs 用于计算梯度的样本
# observable param_circuit所使用的可观测量

def getAdvGrad(param_circuit, inputs, observable):
    n_qubits = param_circuit.num_qubits
    inputqc = QuantumCircuit(n_qubits + 1)

    target_state = inputs / np.linalg.norm(inputs)
    target_state = np.array(target_state, dtype=np.float64)
    _EPS = 1e-10
    while not math.isclose(sum(np.absolute(target_state) ** 2), 1.0, abs_tol=_EPS):
        norm = np.sqrt(sum(np.abs(target_state) ** 2))
        target_state = target_state / norm

    print("target_state", type(target_state))
    controlled_prepare = StatePreparation(target_state).control()

    inputqc.h(0)
    inputqc.append(controlled_prepare, range(n_qubits + 1))

    grad_list = []

    for index, datapoint in enumerate(inputs):
        qc = deepcopy(inputqc)
        grad_circuit = getGrad(n_qubits + 1, index)
        qc.x(0)
        qc.append(grad_circuit, range(n_qubits + 1))
        qc.x(0)

        qc.append(param_circuit, range(1, n_qubits + 1))
        qc.cz(control_qubit=0, target_qubit=1)
        qc.h(0)

        qc.save_expectation_value(quantum_info.Pauli(observable), [0], observable)

        transpiled_qc = transpile(qc, Aer.get_backend('qasm_simulator'))
        job = Aer.get_backend('qasm_simulator').run(transpiled_qc)
        grad_list.append(job.result().data()[observable])
    return grad_list

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
    target_state = np.array(target_state, dtype=np.float64)
    controlled_prepare = StatePreparation(target_state).control()

    input_qc.h(0)
    input_qc.append(controlled_prepare, range(n_qubits + 1))

    grad_list = []
    estimator = Estimator()
    for index, _ in enumerate(inputs):
        qc = deepcopy(input_qc)
        qc.barrier()
        grad_circuit = get_grad_circuit(n_qubits + 1, index)
        qc.x(0)
        qc.barrier()
        qc.compose(grad_circuit, inplace=True)
        qc.barrier()
        qc.x(0)
        qc.barrier()
        qc.compose(param_circuit, qubits=range(1, n_qubits+1), inplace=True)
        qc.barrier()
        qc.cz(control_qubit=0, target_qubit=1)
        qc.barrier()
        qc.h(0)
        qc.barrier()
        print(qc)
        lis = [("IIZ", 1)]
        observable = SparsePauliOp.from_list(lis)
        job = estimator.run(qc, observable)
        print("job.result()", job.result())
        grad_list.append(job.result().values[0])

    return grad_list



if __name__ == '__main__':
    getUtheta()
    ansat = RealAmplitude(2, 1)
    observable = TwoClassifyObservable(2)
    model1 = AmplitudeModel(2, ansat, observable)
    model1.param = np.array([np.pi, np.pi, np.pi, np.pi])

    data = torch.tensor(np.array([[0.5, 0.5, 0.5, 0.5]]))

    import matplotlib.pyplot as plt

    start_exponent = -1
    end_exponent = -20

    values = []
    differences = []

    for exponent in range(start_exponent, end_exponent-1, -1):
        value = 10 ** exponent
        res1 = get_adv_grad(model1.get_fixed_ansatz_circuit(), data[0], model1.observables)
        res2 = getAdvGrad(model1.get_fixed_ansatz_circuit(), data[0], "Z")
        res3 = model1.evaluate_encoding_gradient(data, 1, value)[0].reshape(-1)
        print(res1)
        print(res2)
        print(res3)
        absolute_difference = abs(res1-res3)

        values.append(value)
        differences.append(np.sum(absolute_difference))

        print(value, np.sum(absolute_difference))

    plt.figure(figsize=(12, 8))
    plt.plot(values, differences, marker='o', linestyle='-', color='b')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Value (10^exponent)')
    plt.ylabel('Absolute Difference Sum')
    plt.title('Absolute Difference Sum vs Value')

    exponents = range(start_exponent, end_exponent - 1, -1)
    xticks = [10 ** exp for exp in exponents]
    xtick_labels = [f'1e{exp}' for exp in exponents]
    plt.xticks(xticks, xtick_labels, rotation=45)

    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.show()
    print(model1.circuit)




# if __name__ == '__main__':
#     # 获取数据源，dataloader
#     n_test_samples = 64
#     batch_size = 1
#     X_test = datasets.MNIST(root='../data', train=False, download=True,
#                             transform=transforms.Compose([transforms.Resize([16, 16]), transforms.ToTensor()]))
#
#     idx = np.concatenate(
#         (np.where(X_test.targets == 0)[0][:n_test_samples], np.where(X_test.targets == 1)[0][:n_test_samples]))
#     X_test.data = X_test.data[idx]
#     X_test.targets = X_test.targets[idx]
#     test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=True)
#     data_iter = iter(test_loader)
#     inputs, labels = next(data_iter)
#     param_circuit = getUtheta()
#     #
#     print(param_circuit)
#     print(type(param_circuit), type(inputs))
#     res = getAdvExample(param_circuit, inputs)
#     print(res)
