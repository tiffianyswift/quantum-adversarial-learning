# deepfool
import numpy as np
import torch


class DeepFoolAttack:
    def __init__(self, epsilon=None, epsilons=None, iters=50):
        self.iters = iters
        if epsilon is None and epsilons is None:
            raise ValueError("epsilon和spsilons不能都为None")
        self.epsilon = epsilon
        self.epsilons = epsilons
        if self.epsilon is None:
            self.epsilon = epsilons[0]

    def generate_adv_example(self, model, examples, labels):
        adv_examples = examples.clone()
        activation_value, label = model.predict(examples)
        iteration = 0

        while iteration < self.iters:
            data_grad = model.get_input_grad(examples, labels)
            # For a linear targetmodel and L2 norm, the perturbation can be directly derived
            # Her, for simplicity, let's just use a scaled version of the gradient
            norm_grad = torch.norm(data_grad)
            perturbation = data_grad / norm_grad * self.epsilon

            perturbation = perturbation.reshape(examples.shape)
            adv_examples = adv_examples + perturbation

            adv_examples = np.clip(adv_examples, 0, 1)

            _, current_label = model.predict(adv_examples)
            print(iteration)
            print(current_label)
            print(label)
            if current_label != label:
                break
            iteration += 1
        return adv_examples


    def batch_generate_adv_example(self, model, examples, labels):
        adv_examples = torch.empty(len(self.epsilons), *examples.shape)
        original_epsilon = self.epsilon
        for idx, epsilon in enumerate(self.epsilons):
            self.epsilon = epsilon
            adv_examples[idx] = self.generate_adv_example(model, examples, labels)
        self.epsilon = original_epsilon
        return adv_examples