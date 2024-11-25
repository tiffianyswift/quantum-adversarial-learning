#
import numpy as np


class DeepFoolAttack:
    def __init__(self, epsilon=1e-6, iters=50):
        self.epsilon = epsilon
        self.iters = iters

    def generate_adv_examples(self, model, example):
        original_example = example.copy()
        label = model.predict(example)
        iteration = 0

        while iteration < self.iters:
            data_grad = model.get_gradient(example)
            perturbation = self._compute_perturbation(data_grad)
            example += perturbation

            current_label = model.predict(example)
            if current_label != label:
                break
            iteration += 1

        adv_example = np.clip(example, 0, 1)
        return adv_example

    def _compute_perturbation(self, data_grad):
        # For a linear targetmodel and L2 norm, the perturbation can be directly derived
        # Her, for simplicity, let's just use a scaled version of the gradient
        norm_grad = np.linalg.norm(data_grad)
        perturbation = data_grad / norm_grad * self.epsilon

        return perturbation
