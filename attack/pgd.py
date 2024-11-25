# Projected Gradient Descent
import numpy as np
class PGDAttack:
    def __init__(self, epsilon=0.1, alpha=0.01, iters=10):
        self.epsilon = epsilon
        self.alpha = alpha
        self.iters = iters

    def generate_adv_example(self, model, example):
        original_example = example.copy()
        # 随机初始点
        perturbed_example = example.copy() + np.random.randn(example.shape)*self.epsilon
        for _ in range(self.iters):
            data_grad = model.get_gradient(perturbed_example)
            sign_data_grad = np.sign(data_grad)
            perturbed_example = perturbed_example + self.alpha * sign_data_grad

            # 投影
            perturbation = np.clip(perturbed_example - original_example, -self.epsilon, self.epsilon)
            perturbed_example = np.clip(original_example + perturbation, 0, 1)

        return perturbed_example