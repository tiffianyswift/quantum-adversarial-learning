# Momentum Iterative Method
import numpy as np


class MIMAttack:
    def __init__(self, epsilon=0.1, decay_rate=1.0, iters=10):
        """
        param epsilon:
        para decay_rate:
        param iters:
        """
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.iters = iters

    def generate_adv_example(self, model, example):
        momentum = np.zeros_like(example)
        adv_example = np.copy(example)

        for _ in range(self.iters):
            data_grad = model.get_gradient(adv_example)
            momentum = self.decay_rate * momentum + data_grad / np.linalg.norm(data_grad, ord=1)
            adv_temp = adv_example + self.epsilon * np.sign(momentum)
            adv_example = np.clip(adv_temp, 0, 1)

        return adv_example
