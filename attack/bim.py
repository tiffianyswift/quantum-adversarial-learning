# Basic Iterative Method
import numpy as np


class BIMAttack:
    def __init__(self, epsilon=0.1, alpha=0.01, iters=10):
        """
        :param epsilon: 攻击最大扰动
        :param alpha: 每步攻击的步长
        :param iters: 迭代次数
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.iters = iters

    def generate_adv_example(self, model, example):
        original_example = example.copy()
        perturbed_example = example.copy()
        for _ in range(self.iters):
            data_grad = model.get_gradient(perturbed_example)
            sign_data_grad = np.sign(data_grad)

            perturbed_example = perturbed_example + self.alpha * sign_data_grad

            perturbation = np.clip(perturbed_example - original_example, -self.epsilon, self.epsilon)

            perturbed_example = np.clip(original_example + perturbation, 0, 1)

        return perturbed_example
