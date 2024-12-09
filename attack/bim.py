# Basic Iterative Method
import numpy as np
import torch


class BIMAttack:
    def __init__(self, epsilon=None, epsilons=None, alpha=0.1, iters=10):
        """
        :param epsilon: 攻击最大扰动
        :param alpha: 每步攻击的步长
        :param iters: 迭代次数
        """
        self.alpha = alpha
        self.iters = iters
        if epsilon is None and epsilons is None:
            raise ValueError("epsilon 和 epsilons不能都为None")
        self.epsilon = epsilon
        self.epsilons = epsilons
        if self.epsilon is None:
            self.epsilon = epsilons[0]

    def generate_adv_example(self, model, examples, labels):
        original_examples = examples.clone()
        perturbed_examples = examples.clone()
        for _ in range(self.iters):
            examples_gradient = model.get_input_grad(perturbed_examples, labels)
            data_grad = examples_gradient.reshape(examples.shape)
            sign_data_grad = torch.sign(data_grad)
            perturbed_examples = perturbed_examples + self.alpha * sign_data_grad
            perturbation = torch.clamp(perturbed_examples - original_examples, -self.epsilon, self.epsilon)
            perturbed_examples = torch.clamp(original_examples + perturbation, 0, 1)

        return perturbed_examples

    def batch_generate_adv_example(self, model, examples, labels):
        adv_examples = torch.empty(len(self.epsilons), *examples.shape)
        original_epsilon = self.epsilon
        for idx, epsilon in enumerate(self.epsilons):
            self.epsilon = epsilon
            adv_examples[idx] = self.generate_adv_example(model, examples, labels)

        self.epsilon = original_epsilon
        return adv_examples

