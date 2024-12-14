import numpy as np
import pickle
import os

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale


def get_dataloader(dataset_name, dataset_type, dataset_code, per_class_size, batch_size, shuffle=False):
    assert dataset_name in ['mnist', 'iris', 'kdd-cup'], f"'{dataset_name}' is not dataset name"
    assert dataset_type in ['train', 'test'], f"'{dataset_type}' is not support"
    assert dataset_code in ['default'], f"'{dataset_code}' is not support"
    script_dir = os.path.dirname(os.path.realpath(__file__))
    if dataset_name == 'mnist' and dataset_type == 'train' and dataset_code == 'default':
        x_train = datasets.MNIST(root=script_dir+'/../data', train=True, download=True,
                                 transform=transforms.Compose([transforms.Resize([16, 16]), transforms.ToTensor()]))

        idx = np.concatenate(
            (np.where(x_train.targets == 0)[0][:per_class_size], np.where(x_train.targets == 1)[0][:per_class_size])
        )
        x_train.data = x_train.data[idx]
        x_train.targets = x_train.targets[idx]
        train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=shuffle)
        return train_loader
    elif dataset_name == 'mnist' and dataset_type == 'test' and dataset_code == 'default':
        x_test = datasets.MNIST(root=script_dir+'/../data', train=False, download=True,
                                transform=transforms.Compose([transforms.Resize([16, 16]), transforms.ToTensor()]))
        idx = np.concatenate(
            (np.where(x_test.targets == 0)[0][:per_class_size], np.where(x_test.targets == 1)[0][:per_class_size])
        )
        x_test.data = x_test.data[idx]
        x_test.targets = x_test.targets[idx]
        test_loader = torch.utils.data.DataLoader(x_test, batch_size=batch_size, shuffle=shuffle)
        return test_loader
    elif dataset_name == 'iris' and dataset_type == 'train' and dataset_code == 'default':
        # 使用Scikit-Learn加载Iris数据集
        iris = load_iris()
        x, y = iris.data, iris.target

        x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42)

        x_train = minmax_scale(x_train, feature_range=(0, np.pi))

        # 创建一个空列表来保存抽取的样本
        x_train_samples = []
        y_train_samples = []

        # 按类别抽样
        for class_label in np.unique(y_train):
            # 获取当前类别的样本
            class_mask = (y_train == class_label)
            x_class = x_train[class_mask]
            y_class = y_train[class_mask]

            # 如果类别样本数大于 per_class_size，进行随机抽样
            if len(x_class) >= per_class_size:
                indices = np.random.choice(len(x_class), per_class_size, replace=False)
                x_train_samples.append(x_class[indices])
                y_train_samples.append(y_class[indices])
            else:
                # 如果类别样本数小于 per_class_size，就直接全部取
                x_train_samples.append(x_class)
                y_train_samples.append(y_class)

        # 将抽取的所有样本合并
        x_train_samples = np.vstack(x_train_samples)
        y_train_samples = np.hstack(y_train_samples)

        x_train_tensor = torch.tensor(x_train_samples, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_samples, dtype=torch.int64)

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        return train_loader

    elif dataset_name == 'iris' and dataset_type == 'test' and dataset_code == 'default':
        iris = load_iris()
        x, y = iris.data, iris.target

        _, x_test, _, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        x_test = minmax_scale(x_test, feature_range=(0, np.pi))

        # 创建一个空列表来保存抽取的样本
        x_test_samples = []
        y_test_samples = []

        # 按类别抽样
        for class_label in np.unique(y_test):
            # 获取当前类别的样本
            class_mask = (y_test == class_label)
            x_class = x_test[class_mask]
            y_class = y_test[class_mask]

            # 如果类别样本数大于 per_class_size，进行随机抽样
            if len(x_class) >= per_class_size:
                indices = np.random.choice(len(x_class), per_class_size, replace=False)
                x_test_samples.append(x_class[indices])
                y_test_samples.append(y_class[indices])
            else:
                # 如果类别样本数小于 per_class_size，就直接全部取
                x_test_samples.append(x_class)
                y_test_samples.append(y_class)

        # 将抽取的所有样本合并
        x_test_samples = np.vstack(x_test_samples)
        y_test_samples = np.hstack(y_test_samples)

        x_test_tensor = torch.tensor(x_test_samples, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_samples, dtype=torch.int64)

        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

        return test_loader

    elif dataset_name == 'cup99' and dataset_type == 'train' and dataset_code == 'default':
        file_path = script_dir + '/../data/cup99/data.pickle'

        with open(file_path, 'rb') as file:
            data_o = np.array(pickle.load(file))

        # 获取每个类别的数据
        data_p = data_o[np.where(data_o[:, -1] == 0)][0:500]
        data_n = data_o[np.where(data_o[:, -1] == 1)][0:500]

        data_pn_without_label = np.concatenate([data_p, data_n])[:, 0:5]
        minv = data_pn_without_label.min(axis=0)
        maxv = data_pn_without_label.max(axis=0)
        avg = maxv - minv

        # 每类选取指定数量的样本
        train_p = data_p[:per_class_size] if len(data_p) >= per_class_size else data_p
        train_n = data_n[:per_class_size] if len(data_n) >= per_class_size else data_n

        train_x = torch.Tensor(((np.concatenate([train_p, train_n])[:, :5]) - minv) / avg)
        train_y = torch.Tensor(np.concatenate([train_p, train_n])[:, -1])

        train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        return train_loader

    elif dataset_name == 'cup99' and dataset_type == 'test' and dataset_code == 'default':
        file_path = script_dir + '/../data/cup99/data.pickle'

        with open(file_path, 'rb') as file:
            data_o = np.array(pickle.load(file))

        # 获取每个类别的数据
        data_p = data_o[np.where(data_o[:, -1] == 0)][0:500]
        data_n = data_o[np.where(data_o[:, -1] == 1)][0:500]

        data_pn_without_label = np.concatenate([data_p, data_n])[:, 0:5]
        minv = data_pn_without_label.min(axis=0)
        maxv = data_pn_without_label.max(axis=0)
        avg = maxv - minv

        # 每类选取指定数量的样本
        test_p = data_p[-per_class_size:] if len(data_p) >= per_class_size else data_p
        test_n = data_n[-per_class_size:] if len(data_n) >= per_class_size else data_n

        test_x = torch.Tensor(((np.concatenate([test_p, test_n])[:, :5]) - minv) / avg)
        test_y = torch.Tensor(np.concatenate([test_p, test_n])[:, -1])

        test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

        return test_loader

