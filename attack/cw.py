# Carlini & Wagner Attack
import torch.nn.functional as F
import torch


class CWAttack:
    def __init__(self, kappa=0, learning_rate=0.01, iters=100, initial_const=0.01, targeted=False):
        """
        :param kappa: Confidence parameter, controls the strength of adversarial examples
        :param learning_rate: Learning rate for gradient descent optimization
        :param iter: Maximum number of iterations for optimization
        :param initial_const: Initial value of the constant for optimization
        """
        self.kappa = kappa
        self.learning_rate = learning_rate
        self.iters = iters
        self.initial_const = initial_const
        self.targeted = targeted
        self.learning_rates = [0.1, 0.01, 0.001]

    def generate_adv_example(self, model, examples, labels):
        original_examples = examples.clone()
        perturbations = torch.empty_like(original_examples).uniform_(-0.001, 0.001)
        targeted_labels = (1 - labels)

        for i in range(self.iters):
            adv_examples = examples + perturbations
            adv_examples = torch.clamp(adv_examples, 0, 1)
            activations, current_labels = model.predict(adv_examples)
            activations.requires_grad_()
            losses = targeted_labels - activations
            for loss in losses:
                loss.backward()

            data_grad = activations.grad.reshape(-1, 1) * model.get_input_grad(adv_examples, current_labels)

            perturbations = perturbations - self.learning_rate * data_grad.reshape(original_examples.shape).sign()
            print("idx", i)

        adv_examples = examples + perturbations
        return adv_examples

    def batch_generate_adv_example(self, model, examples, labels):
        adv_examples = torch.empty(len(self.learning_rates), *examples.shape)
        original_learning_rate = self.learning_rate
        for idx, learning_rate in enumerate(self.learning_rates):
            self.learning_rate = learning_rate
            adv_examples[idx] = self.generate_adv_example(model, examples, labels)
        self.learning_rate = original_learning_rate
        return adv_examples
