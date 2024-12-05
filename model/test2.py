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
        # print(qc)
        lis = [("IIZ", 2)]
        observable = SparsePauliOp.from_list(lis)
        job = estimator.run(qc, observable)
        # print("job.result()", job.result())
        grad_list.append(job.result().values[0])

    return grad_list



if __name__ == '__main__':
    ansat = RealAmplitude(2, 2)
    observable = TwoClassifyObservable(2)
    model1 = AmplitudeModel(2, ansat, observable)
    model1.param = np.random.randn(4)

    data = torch.tensor(np.array([[0, 1/np.sqrt(2), 1/np.sqrt(2), 0]]))

    import matplotlib.pyplot as plt

    start_exponent = -1
    end_exponent = -20

    values = []
    differences = []

    for exponent in range(start_exponent, end_exponent-1, -1):
        value = 10 ** exponent
        res1 = get_adv_grad(model1.get_fixed_ansatz_circuit(), data[0], model1.observables)
        res3 = model1.evaluate_encoding_gradient(data, 1, value)[0].reshape(-1)
        print(res1)
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
