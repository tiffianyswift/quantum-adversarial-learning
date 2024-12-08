from qiskit import QuantumCircuit
from qiskit.extensions import UnitaryGate
from qiskit_machine_learning.circuit.library import RawFeatureVector

import numpy as np

from data_preprocess.dataloader import get_dataloader
from ansatz import RealAmplitude
from model import AmplitudeModel
from observable import TwoClassifyObservable
import torch

if __name__ == '__main__':
    batch_size = 4
    n_qubits = 8
    reps = 3
    ansatz = RealAmplitude(8, 3)
    observable = TwoClassifyObservable(8)
    target_model = AmplitudeModel(8, ansatz, observable)
    target_model.load(path="../model_saved/param.npy")

    dataloader = get_dataloader('mnist', 'train', 'default', 8, batch_size , True)
    loss_func = torch.nn.BCELoss()

    batch = next(iter(dataloader))

    inputs, labels = batch

    pred = torch.tensor(target_model(inputs), dtype=torch.float32, requires_grad=True)

    loss = loss_func(pred, labels.float())
    loss.backward()

    loss_grad = pred.grad.reshape(batch_size, 1)
    ansatz_grad = torch.tensor(target_model.solve_ansatz_gradient(inputs))
    batch_ansatz_grad = torch.sum(loss_grad*ansatz_grad, dim=0)

    print(loss_grad.shape)
    print(ansatz_grad.shape)

    print(loss_grad)
    print(ansatz_grad)

    print(loss_grad*ansatz_grad)

    print(batch_ansatz_grad)

    # print(grad0.view(4, 1) * grad1)