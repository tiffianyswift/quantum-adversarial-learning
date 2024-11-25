# Carlini & Wagner Attack
import numpy as np


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
        self.initial_const = 0.01
        self.targeted = False

    def generate_adv_example_with_logits(self, model, example):
        original_example = example.copy()
        w = np.zeros_like(example)
        const = self.initial_const
        perturbation = np.random.uniform(-0.001, 0.001, example.shape)
        label = model.predict(example)

        targeted_label = (label + np.random.randint(1, 10)) % 10

        for _ in range(self.iters):
            adv_image = example + perturbation
            adv_image = np.clip(adv_image, 0, 1)

            # 假设为多分类，返回的是softmax后的值
            logits = model.predict(adv_image)

            target_logits = logits[0][targeted_label]
            other_logits = np.concatenate([logits[:targeted_label], logits[targeted_label + 1:]])

            max_other_logits = np.max(other_logits)

            if self.targeted:
                # 如果是定向攻击，需要让目标类别的logits最大
                loss = max_other_logits - target_logits + self.kappa
            else:
                # 如果是非定向攻击，需要让正确类别的logits不是最大
                loss = target_logits - max_other_logits + self.kappa

            # Gradient of the loss function with respect to the input
            grad = np.gradient(loss, adv_image)

            perturbation = perturbation + self.learning_rate * grad

        adversarial_example = example + perturbation
        return adversarial_example

    def generate_adv_example(self, model, example):
        perturbation = np.random.uniform(-0.001, 0.001, example.shape)
        original_label = model.predict(example)

        for _ in range(self.iters):
            adv_image = example + perturbation
            adv_image = np.clip(adv_image, 0, 1)

            # 假设为多分类
            logits = model.predict_with_logits(adv_image)

            original_logit = logits[original_label]
            other_logits = np.concatenate([logits[:original_label], logits[original_label + 1:]])
            max_other_logits = np.max(other_logits)

            if self.targeted:
                # 如果是定向攻击，需要让目标类别的logits最大
                loss = max_other_logits - original_logit + self.kappa
            else:
                loss = original_logit - max_other_logits + self.kappa

            perturbation_grad = np.gradient(loss, perturbation)
            perturbation = perturbation + self.learning_rate * perturbation_grad

        adversarial_example = example + perturbation
        return adversarial_example