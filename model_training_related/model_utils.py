from model_training_related.ansatz import RealAmplitude, FakeAmplitude
from model_training_related.encoding import AmplitudeEncoding
from model_training_related.model import AmplitudeModel
from model_training_related.observable import TwoClassifyObservable
from data_preprocess.dataloader import get_dataloader

import numpy as np

from torch.optim import Adam
import torch


def get_predict_label(output, loss_func_name):
    if loss_func_name == 'bce':
        return np.round(output)
    else:
        return None


def get_model(n_qubits, rep, ansatz_type, observable_type, encoding_type, param_load_path):
    if observable_type == 'two_classify':
        observable = TwoClassifyObservable(n_qubits=n_qubits)
    else:
        return None

    if ansatz_type == 'real_amplitude':
        ansatz = RealAmplitude(n_qubits, rep)
    else:
        return None

    if encoding_type == 'amplitude':
        model = AmplitudeModel(n_qubits, ansatz, observable)
    else:
        return None

    if len(param_load_path) != 0:
        model.load(param_load_path)

    return model


def evaluate(mp):
    # circuit hyperparam
    assert 'n_qubits' in mp, "n_qubits为空"
    assert 'rep' in mp, "n_qubits为空"
    assert 'ansatz_type' in mp, "n_qubits为空"
    assert 'observable_type' in mp, "n_qubits为空"
    assert 'encoding_type' in mp, "n_qubits为空"

    assert 'loss_func_name' in mp, "optimizer"

    assert 'test_param_load_path' in mp, "optimizer"

    assert 'dataset_name' in mp, "optimizer"
    assert 'test_dataset_type' in mp, "optimizer"
    assert 'dataset_code' in mp, "optimizer"
    assert 'n_test_samples' in mp, "n_qubits为空"
    assert 'test_batch_size' in mp, "n_qubits为空"
    assert 'test_shuffle' in mp, 'n_qubits'

    n_qubits = mp['n_qubits']
    rep = mp['rep']
    ansatz_type = mp['ansatz_type']
    observable_type = mp['observable_type']
    encoding_type = mp['encoding_type']
    param_load_path = mp['test_param_load_path']

    # optimizer hyperparam
    loss_func_name = mp['loss_func_name']

    # setting
    dataset_name = mp['dataset_name']
    test_dataset_type = mp['test_dataset_type']
    dataset_code = mp['dataset_code']
    n_test_samples = mp['n_test_samples']
    test_batch_size = mp['test_batch_size']
    test_shuffle = mp['test_shuffle']

    model = get_model(n_qubits, rep, ansatz_type, observable_type, encoding_type, param_load_path)
    if model is None:
        return

    test_loader = get_dataloader(dataset_name=dataset_name, dataset_type=test_dataset_type, dataset_code=dataset_code,
                                 per_class_size=n_test_samples, batch_size=test_batch_size, shuffle=test_shuffle)

    for batch_idx, (data, target) in enumerate(test_loader):
        model_output = model(data)
        target_predict = get_predict_label(model_output, loss_func_name)
        print(target_predict, target)


def train(mp):
    # circuit hyperparam
    assert 'n_qubits' in mp, "n_qubits为空"
    assert 'rep' in mp, "n_qubits为空"
    assert 'ansatz_type' in mp, "n_qubits为空"
    assert 'observable_type' in mp, "n_qubits为空"
    assert 'encoding_type' in mp, "n_qubits为空"

    assert 'n_train_samples' in mp, "n_qubits为空"
    assert 'train_batch_size' in mp, "n_qubits为空"

    assert 'optimizer_type' in mp, "optimizer"
    assert 'lr' in mp, "optimizer"
    assert 'epoch' in mp, "optimizer"
    assert 'loss_func_name' in mp, "optimizer"

    assert 'train_param_load_path' in mp,  "optimizer"
    assert 'param_dump_path' in mp, "optimizer"

    assert 'dataset_name' in mp, "optimizer"
    assert 'train_dataset_type' in mp, "optimizer"
    assert 'dataset_code' in mp, "optimizer"

    n_qubits = mp['n_qubits']
    rep = mp['rep']
    ansatz_type = mp['ansatz_type']
    observable_type = mp['observable_type']
    encoding_type = mp['encoding_type']
    train_param_load_path = mp['train_param_load_path']
    param_dump_path = mp['param_dump_path']

    # dataset hyperparam
    n_train_samples = mp['n_train_samples']
    train_batch_size = mp['train_batch_size']

    # optimizer hyperparam
    optimizer_type = mp['optimizer_type']
    lr = mp['lr']
    epoch = mp['epoch']
    loss_func_name = mp['loss_func_name']

    # setting
    dataset_name = mp['dataset_name']
    train_dataset_type = mp['train_dataset_type']
    dataset_code = mp['dataset_code']

    model = get_model(n_qubits, rep, ansatz_type, observable_type, encoding_type, train_param_load_path)
    if model is None:
        return

    torch_param = torch.tensor(model.param, dtype=torch.float32, requires_grad=True)
    if optimizer_type == 'adam':
        optimizer = Adam([torch_param], lr=lr)
    else:
        return

    train_loader = get_dataloader(dataset_name=dataset_name, dataset_type=train_dataset_type, dataset_code=dataset_code,
                                 per_class_size=n_train_samples, batch_size=train_batch_size)

    if loss_func_name == 'bce':
        loss_func = torch.nn.BCELoss()
    else:
        return

    for _ in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = torch.tensor(model(data), dtype=torch.float32, requires_grad=True)

            loss = loss_func(output, target.float())
            loss.backward()

            loss_grad = output.grad.reshape(train_batch_size, 1)
            ansatz_grad = torch.tensor(model.solve_ansatz_gradient(data))
            batch_ansatz_grad = torch.sum(loss_grad*ansatz_grad, dim=0)

            torch_param.grad = batch_ansatz_grad
            optimizer.step()
            model.param = torch_param.detach().numpy()
            model.dump(param_dump_path)
            print(loss)


if __name__ == '__main__':
    train_loader = get_dataloader('iris', 'train', 'default', 8, 4, True)
    for batch_idx, (data, target) in enumerate(train_loader):
        print(batch_idx, data, target)
