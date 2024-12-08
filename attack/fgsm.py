# Fast Gradient Sign Method
import numpy as np
import torch


# mnist 建议值 0.15
class FGSMAttack:
    def __init__(self, epsilon=None, epsilons=None):
        if epsilon is None and epsilons is None:
            raise ValueError("epsilon 和 epsilons不能都为None")
        self.epsilon = epsilon
        self.epsilons = epsilons
        if self.epsilon is None:
            self.epsilon = epsilons[0]

    # 使用epsilons生成对抗样本
    def generate_adv_example(self, model, examples, labels, examples_gradient=None):
        if examples_gradient is None:
            examples_gradient = model.get_input_grad(examples, labels)
        data_grad = examples_gradient.reshape(examples.shape)
        sign_data_grad = np.sign(data_grad)
        adv_example = examples + self.epsilon * sign_data_grad
        adv_example = np.clip(adv_example, 0, 1)
        return adv_example

    # 使用epsilons生成对抗样本
    def batch_generate_adv_example(self, model, examples, labels, examples_gradient=None):
        adv_examples = torch.empty(len(self.epsilons), *examples.shape)
        if examples_gradient is None:
            examples_gradient = model.get_input_grad(examples, labels)
        original_epsilon = self.epsilon
        for idx, epsilon in enumerate(self.epsilons):
            self.epsilon = epsilon
            adv_examples[idx] = self.generate_adv_example(model, examples, labels, examples_gradient)

        self.epsilon = original_epsilon
        return adv_examples

