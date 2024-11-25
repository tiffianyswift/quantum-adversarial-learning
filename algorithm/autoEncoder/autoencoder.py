from qiskit import transpile
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
import torch
import numpy as np
from Utils.Ansatz import RealAplitude, FakeAplitude
from Utils.Encoding import AmplitudeEncoding
from qiskit.primitives import Estimator
from Utils.Observable import TwoClassfiyObservable
import qiskit
from Utils.Util import getGrad, saveModel, saveVal, setModelParamFromFile
from Utils.Model import AmplitudeModel
from Utils.Observable import TwoClassfiyObservable
from Utils.Ansatz import RealAplitude
from Utils.Util import getGrad, saveModel, saveVal, setModelParamFromFile

import numpy as np
import datetime

import torch.utils.data
from torchvision import datasets, transforms
from torch.optim import Adam
from qiskit.algorithms.optimizers import COBYLA
from scipy.optimize import minimize


class AutoEncoder:
    def __init__(self, n_qubits, ansatz, observables):
        self.n_qubits = n_qubits
        self.ansatz = ansatz
        self.param = np.random.randn(len(self.ansatz.param))
        self.observables = observables.observable
        self.encoding = AmplitudeEncoding(n_qubits)
        self.circuit = self.encoding.circuit
        self.circuit.compose(self.ansatz.circuit, inplace=True)

        self.helpcircuit = AmplitudeEncoding(n_qubits).circuit

    def __call__(self, x):
        # print(self.param[0])
        # 将两个矩阵扁平化成一维数组
        num_samples = x.shape[0]
        x = x.view(num_samples, -1)
        params = np.broadcast_to(self.param, (num_samples, len(self.param)))

        theta = np.concatenate((x, params), axis=1).reshape(-1)


        qc = self.circuit.assign_parameters(theta)

        backend = qiskit.Aer.get_backend('statevector_simulator')
        t_qc = transpile(qc, backend)
        job = backend.run(t_qc)

        qc2 = self.helpcircuit.assign_parameters(x.numpy().reshape(-1))

        t_qc2 = transpile(qc2, backend)

        job2 = backend.run(t_qc2)

        statevector1 = job.result().get_statevector().data
        statevector2 = job2.result().get_statevector().data

        return -np.real(np.vdot(statevector1, statevector2))




def objfun(weights, *args):
    train_loader = args[0]
    model = args[1]
    model.param = weights
    loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        loss += model(data)
    print(loss)
    return loss




if __name__ == '__main__':
    n_train_samples = 16
    batch_size = 1
    X_train = datasets.MNIST(root='../../data', train=True, download=False,
                             transform=transforms.Compose([transforms.Resize([16, 16]), transforms.ToTensor()]))

    idx = np.concatenate(
        (np.where(X_train.targets == 0)[0][:n_train_samples], np.where(X_train.targets == 1)[0][:n_train_samples])
    )
    X_train.data = X_train.data[idx]
    X_train.targets = X_train.targets[idx]
    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)

    n_test_samples = 64
    X_test = datasets.MNIST(root='../../data', train=False, download=False,
                            transform=transforms.Compose([transforms.Resize([16, 16]), transforms.ToTensor()]))
    idx = np.concatenate(
        (np.where(X_test.targets == 0)[0][:n_test_samples], np.where(X_test.targets == 1)[0][:n_test_samples])
    )
    X_test.data = X_test.data[idx]
    X_test.targets = X_test.targets[idx]
    test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=True)
    # 什么数据，使用MNIST
    data_iter = iter(X_train)

    data, target = next(data_iter)


    n_qubits = 8
    rep = 3
    observable = TwoClassfiyObservable(n_qubits)

    ansatz = FakeAplitude(n_qubits, rep)
    model = AutoEncoder(n_qubits, ansatz, observable)



    # torch_param = torch.tensor(model.param, dtype=torch.float32, requires_grad=True)
    res = minimize(fun=objfun, x0=model.param, args=(train_loader, model),
                   method="COBYLA", options={"maxiter": 2000})
    # optimizer = Adam([torch_param], lr=0.01)




