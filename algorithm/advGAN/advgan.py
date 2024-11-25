import torch.nn as nn
from torchvision.utils import save_image

from Utils.Model import AmplitudeModel, getGrad
from Utils.Ansatz import RealAplitude
from Utils.Observable import TwoClassfiyObservable
from Utils.Util import setModelParamFromFile, saveVal
from qiskit_machine_learning.circuit.library import RawFeatureVector
import time
import os
import copy
import numpy as np
# PyTorch
import torch
from torch.autograd import Function
import torch.nn.functional as F

import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
# Qiskit
import qiskit
from qiskit import transpile,assemble
from qiskit import quantum_info
from qiskit.visualization import *
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import StatePreparation
# Plotting
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256)
        )
    def forward(self, initial_noise):
        noise = self.model(initial_noise)
        return noise

class Discriminator(AmplitudeModel):
    def __init__(self):
        super().__init__(n_qubits=8, ansatz=RealAplitude(8, 3), observables=TwoClassfiyObservable(8))

class AttackedModel(AmplitudeModel):
    def __init__(self):
        super().__init__(n_qubits=8, ansatz=RealAplitude(8, 3), observables=TwoClassfiyObservable(8))
        setModelParamFromFile(self, "../../mnist/mnist01/model_saved/saved 2128")

if __name__ == '__main__':

    n_train_samples = 256
    batch_size = 4
    X_train = datasets.MNIST(root='../data', train=True, download=False,
                             transform=transforms.Compose([transforms.Resize([16, 16]), transforms.ToTensor()]))

    idx = np.concatenate(
        (np.where(X_train.targets == 0)[0][:n_train_samples], np.where(X_train.targets == 1)[0][:n_train_samples])
    )
    X_train.data = X_train.data[idx]
    X_train.targets = X_train.targets[idx]
    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)

    n_test_samples = 64
    X_test = datasets.MNIST(root='../data', train=False, download=False,
                            transform=transforms.Compose([transforms.Resize([16, 16]), transforms.ToTensor()]))
    idx = np.concatenate(
        (np.where(X_test.targets == 0)[0][:n_test_samples], np.where(X_test.targets == 1)[0][:n_test_samples])
    )
    X_test.data = X_test.data[idx]
    X_test.targets = X_test.targets[idx]
    test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=True)

    attackedmodel = AttackedModel()
    generator = Generator()
    discriminator = Discriminator()


    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
    dparam = torch.tensor(discriminator.param, dtype=torch.float32, requires_grad=True)
    optimizer_D = torch.optim.Adam([dparam], lr=0.01, betas=(0.5, 0.999))
    for epoch in range(1, 10):
        loss_D_sum = 0
        loss_G_fake_sum = 0
        loss_perturb_sum = 0
        loss_adv_sum = 0

        for batch, (imgs, labels) in enumerate(train_loader):
            optimizer_D.zero_grad()
            imgs1 = imgs.reshape((batch_size, 1, -1))
            perturbation = generator(imgs1)
            # adv_images = imgs1 + torch.clamp(perturbation, -0.3, 0.3)
            adv_images = imgs1 + perturbation

            imgs2 = adv_images.reshape((batch_size, 1, 16, 16)).detach()
            imgs3 = torch.cat((imgs, imgs2), dim=0)
            save_image(imgs3, "images/%d.png" % batch, nrow=4, normalize=True)
            # print(sum(imgs1))

            # 真实标签训练生成器

            pred_real = torch.tensor(discriminator(imgs1), dtype=torch.float32, requires_grad=True)

            loss_D_real = F.binary_cross_entropy(pred_real, torch.ones(batch_size))
            loss_D_real.backward()

            grad1 = torch.tensor(getGrad(discriminator, imgs1, np.array(pred_real.grad)), dtype=torch.float32)
            dparam.grad = grad1

            # 假标签训练生成器（需要裁切一下吗）
            pred_fake = torch.tensor(discriminator(adv_images.detach()), dtype=torch.float32, requires_grad=True)
            loss_D_fake = F.binary_cross_entropy(pred_fake, torch.zeros(batch_size))

            loss_D_fake.backward()
            dparam.grad += torch.tensor(getGrad(discriminator, adv_images.detach(), np.array(pred_fake.grad)), dtype=torch.float32)


            # 用于统计
            loss_D_GAN = loss_D_real + loss_D_fake
            optimizer_D.step()
            discriminator.param = dparam.detach().numpy()
            saveVal(loss_D_GAN.item())
            print(loss_D_GAN.item())

            #
            # generator.zero_grad()
            # # 生成器想要欺骗判别器，要最小化与真实的图像的差距
            # pred_fake = torch.tensor(discriminator(adv_images.detach()), dtype=torch.float32, requires_grad=True)
            # loss_G_fake = F.binary_cross_entropy(pred_fake, torch.ones(batch_size))
            # loss_G_fake.backward(retain_graph=True)  # 为了保存梯度
            #
            # # 使扰动尽可能的小
            # loss_perturb = torch.mean(torch.norm(perturbation.reshape(batch_size, -1), 2, dim=1))
            # # 获取攻击模型的预测值
            # probs = torch.tensor(attackedmodel(adv_images.detach()), dtype=torch.float32, requires_grad=False)
            # onehot_labels = torch.eye(2)[labels]
            #
            #
            # nprobs = torch.stack((1 - probs, probs), dim=1)
            #
            # # C&W loss function，其目的是最小化正确的类和第二大类的值
            # real = torch.sum(onehot_labels * nprobs, dim=1)
            # other, _ = torch.max((1 - onehot_labels) * nprobs - onehot_labels * 10000, dim=1)
            #
            # loss_adv = torch.max(real - other, torch.zeros(batch_size, 1))
            # loss_adv = torch.sum(loss_adv)
            #
            # adv_lambda = 1
            # pert_lambda = 10
            # # c&w损失加上扰动损失
            # loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            # loss_G.backward()
            # # optimizer_G.step()
            #
            #
            # loss_D_sum += loss_D_GAN.item()
            # loss_G_fake_sum += loss_G_fake.item()
            # loss_perturb_sum += loss_perturb.item()
            # loss_adv_sum += loss_adv.item()
            # print("epoch %d: loss_D: %.3f, loss_G_fake: %.3f, loss_perturb: %.3f, loss_adv: %.3f, " %
            #       (epoch, loss_D_GAN.item(), loss_G.item(), loss_perturb.item(), loss_adv.item()))












