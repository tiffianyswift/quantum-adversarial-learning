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

    def generate_adv_example(self, model, examples):
        original_examples = examples.clone()
        const = self.initial_const
        perturbation = torch.empty_like(original_examples).uniform_(-0.001, 0.001)
        _, label = model.predict(examples)
        targeted_label = int((1-label)[0])

        for i in range(self.iters):
            adv_examples = examples + perturbation
            adv_examples = torch.clamp(adv_examples, 0, 1)
            activation, current_labels = model.predict(adv_examples)
            activation.requires_grad_()

            activations = torch.cat((activation, 1-activation), dim=0)

            target_activation = activations[targeted_label]
            other_activation = activations[1-targeted_label]
            # other_activations = np.concatenate([activations[:targeted_label], activations[targeted_label+1:]])
            # max_other_activation = np.max(other_activations)
            if self.targeted:
                loss1 = F.relu(other_activation - target_activation + self.kappa).mean()
            else:
                loss1 = F.relu(target_activation - other_activation + self.kappa).mean()

            # loss2 = torch.norm(adv_examples - original_examples, p=2)
            # loss = loss1 + const * loss2
            loss = loss1
            loss.backward()

            data_grad = activation.grad * model.get_input_grad(adv_examples, current_labels).reshape(examples.shape)
            if self.targeted:
                perturbation = perturbation + self.learning_rate * data_grad.sign()
            else:
                perturbation = perturbation - self.learning_rate * data_grad.sign()
            print("index", i)

        adv_examples = examples + perturbation
        return adv_examples






    # def generate_adv_example_with_logits(self, model, examples):
    #     original_examples = examples.copy()
    #     w = np.zeros_like(examples)
    #     const = self.initial_const
    #     perturbation = np.random.uniform(-0.001, 0.001, examples.shape)
    #     label = model.predict(examples)
    #
    #     targeted_label = (label + np.random.randint(1, 10)) % 10
    #
    #     for _ in range(self.iters):
    #         adv_image = examples + perturbation
    #         adv_image = np.clip(adv_image, 0, 1)
    #
    #         # 假设为多分类，返回的是softmax后的值
    #         logits = model.predict(adv_image)
    #
    #         target_logits = logits[0][targeted_label]
    #         other_logits = np.concatenate([logits[:targeted_label], logits[targeted_label + 1:]])
    #
    #         max_other_logits = np.max(other_logits)
    #
    #         if self.targeted:
    #             # 如果是定向攻击，需要让目标类别的logits最大
    #             loss = max_other_logits - target_logits + self.kappa
    #         else:
    #             # 如果是非定向攻击，需要让正确类别的logits不是最大
    #             loss = target_logits - max_other_logits + self.kappa
    #
    #         # Gradient of the loss function with respect to the input
    #         grad = np.gradient(loss, adv_image)
    #
    #         perturbation = perturbation + self.learning_rate * grad
    #
    #     adversarial_example = examples + perturbation
    #     return adversarial_example
    #
    # def generate_adv_example(self, model, example):
    #     perturbation = np.random.uniform(-0.001, 0.001, example.shape)
    #     original_label = model.predict(example)
    #
    #     for _ in range(self.iters):
    #         adv_image = example + perturbation
    #         adv_image = np.clip(adv_image, 0, 1)
    #
    #         # 假设为多分类
    #         logits = model.predict_with_logits(adv_image)
    #
    #         original_logit = logits[original_label]
    #         other_logits = np.concatenate([logits[:original_label], logits[original_label + 1:]])
    #         max_other_logits = np.max(other_logits)
    #
    #         if self.targeted:
    #             # 如果是定向攻击，需要让目标类别的logits最大
    #             loss = max_other_logits - original_logit + self.kappa
    #         else:
    #             loss = original_logit - max_other_logits + self.kappa
    #
    #         perturbation_grad = np.gradient(loss, perturbation)
    #         perturbation = perturbation + self.learning_rate * perturbation_grad
    #
    #     adversarial_example = example + perturbation
    #     return adversarial_example