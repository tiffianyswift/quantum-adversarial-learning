import torch
import numpy as np

from ansatz import RealAmplitude
from model import AngleModel, AmplitudeModel
from observable import TwoClassifyObservable
from utils.util import get_grad

from qiskit import QuantumCircuit

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

    qc = getGrad(4, 7)
    print(qc)

# data = torch.tensor()
    #
    # print(model1(data))

