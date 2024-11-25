# Fast Gradient Sign Method
import numpy as np


class FGSMAttack:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def generate_adv_example(self, model, example):
        data_grad = model.get_gradient(example)
        sign_data_grad = np.sign(data_grad)
        adv_example = example + self.epsilon * sign_data_grad
        adv_example = np.clip(adv_example, 0, 1)
        return adv_example
