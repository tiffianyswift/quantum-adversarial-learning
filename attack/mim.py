# Momentum Iterative Method
import numpy as np
import torch


class MIMAttack:
    # mnist 建议值 0.15
    def __init__(self, epsilon=None, epsilons=None, decay_rate=1.0, iters=10):
        self.decay_rate = decay_rate
        self.iters = iters
        if epsilon is None and epsilons is None:
            raise ValueError("epsilon 和 epsilons不能都为None")
        self.epsilon = epsilon
        self.epsilons = epsilons
        if self.epsilon is None:
            self.epsilon = epsilons[0]

    # 使用epsilons生成对抗样本
    def generate_adv_example(self, model, examples, labels):
        momentum = torch.zeros_like(examples)
        adv_example = torch.clone(examples)

        for _ in range(self.iters):
            data_grad = model.get_input_grad(examples, labels)
            data_grad = data_grad.reshape(examples.shape)
            momentum = self.decay_rate * momentum + data_grad / torch.norm(data_grad, p=1)
            adv_example = adv_example + self.epsilon * torch.sign(momentum)
            adv_example = torch.clamp(adv_example, 0, 1)

        return adv_example

    # 使用epsilons生成对抗样本
    def batch_generate_adv_example(self, model, examples, labels):
        adv_examples = torch.empty(len(self.epsilons), *examples.shape)
        original_epsilon = self.epsilon
        for idx, epsilon in enumerate(self.epsilons):
            self.epsilon = epsilon
            adv_examples[idx] = self.generate_adv_example(model, examples, labels)

        self.epsilon = original_epsilon
        return adv_examples


