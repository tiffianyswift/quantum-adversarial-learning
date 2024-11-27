import qiskit
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
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
    param = model.state_dict()['q_params']
    data_param = torch.reshape(param, (3, q_depth + 1, n_qubits)).cpu().numpy()
    circuit = QuantumCircuit(n_qubits, name="U(theta)")

    for j in range(q_depth):
        for k in range(n_qubits):
            circuit.h(k)
            circuit.u(data_param[0][j][k], data_param[1][j][k], data_param[2][j][k], k)
#
        for m in range(0, n_qubits):
            circuit.cnot(m, (m + 1) % (n_qubits))
        circuit.barrier()
    for qubit in range(n_qubits):
        circuit.u(data_param[0][q_depth][qubit],
                  data_param[1][q_depth][qubit],
                  data_param[2][q_depth][qubit], qubit)
    return circuit

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

        qc.append(param_circuit, range(1, n_qubits + 1))
        qc.cz(control_qubit=0, target_qubit=1)
        qc.h(0)

        qc.save_expectation_value(quantum_info.Pauli(observable), [0], observable)

        transpiled_qc = transpile(qc, Aer.get_backend('qasm_simulator'))
        job = Aer.get_backend('qasm_simulator').run(transpiled_qc)
        grad_list.append(job.result().data()[observable])
    return grad_list

# 参数1 模型    type: qiskit类型的量子线路，2分类,不包含初态制备
# 参数2 可观测量  type：模型使用的可观测量，应该为单比特
# 参数3  数据   type: 数据，其维度应该小于2**n_qubits
# 应该传入一个二分类分类器
def getAdvExample(param_circuit, data, observable="Z"):
    n_qubits = param_circuit.num_qubits
    inputs = np.array(data).flatten()

    grad_k = getAdvGrad(param_circuit, inputs, observable)
    target_state = np.array(inputs, dtype=np.float64) / np.linalg.norm(inputs)
    _EPS = 1e-10

    while not math.isclose(sum(np.absolute(target_state) ** 2), 1.0, abs_tol=_EPS):
        norm = np.sqrt(sum(np.abs(target_state) ** 2))
        target_state = target_state / norm


    # 初态制备
    predict_circuit = QuantumCircuit(n_qubits)
    state_pre = StatePreparation(target_state)
    predict_circuit.append(state_pre, range(0, n_qubits))
    predict_circuit.append(param_circuit, range(0, n_qubits))
    predict_circuit.save_expectation_value(quantum_info.Pauli(observable), [0], observable)
    transpiled_qc = transpile(predict_circuit, Aer.get_backend('qasm_simulator'))
    job = Aer.get_backend('qasm_simulator').run(transpiled_qc)
    y_predict = job.result().data()[observable]

    # 对抗样本
    adv_example = target_state + 1 * (y_predict - labels.item()) * np.array(grad_k)

    return adv_example



# if __name__ == '__main__':
#     ansat = RealAmplitude(4, 2)
#     observable = TwoClassifyObservable(4)
#     model1 = AmplitudeModel(4, ansat, observable)
#
#     data = torch.tensor(np.random.randn(1, 16))
#     print(type(model1.ansatz.circuit), type(data[0]))
#     circuit = QuantumCircuit(4)
#     qc = getAdvExample(circuit, data[0])
#     print(qc)



if __name__ == '__main__':
    # 获取数据源，dataloader
    n_test_samples = 64
    batch_size = 1
    X_test = datasets.MNIST(root='../data', train=False, download=True,
                            transform=transforms.Compose([transforms.Resize([16, 16]), transforms.ToTensor()]))

    idx = np.concatenate(
        (np.where(X_test.targets == 0)[0][:n_test_samples], np.where(X_test.targets == 1)[0][:n_test_samples]))
    X_test.data = X_test.data[idx]
    X_test.targets = X_test.targets[idx]
    test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=True)
    data_iter = iter(test_loader)
    inputs, labels = next(data_iter)
    param_circuit = getUtheta()
    #
    print(param_circuit)
    print(type(param_circuit), type(inputs))
    res = getAdvExample(param_circuit, inputs)
    print(res)
