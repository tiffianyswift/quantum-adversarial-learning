from attack.fgsm import FGSMAttack
from attack.mim import MIMAttack
from attack.pgd import PGDAttack
from data_preprocess.dataloader import get_dataloader
from model_training_related.ansatz import RealAmplitude
from model_training_related.model import AmplitudeModel
from model_training_related.observable import TwoClassifyObservable
import torch

from target_model.mnist_amplitude_model import MnistAmplitudeModel
from utils.data_plot import plot_mnist, plot_mnist_batch, plot_mnist_batch_classes

if __name__ == '__main__':
    target_model = MnistAmplitudeModel()
    batch_size = 1
    epsilons = [0.15, 0.1, 0.01, 0.001]

    dataloader = get_dataloader('mnist', 'train', 'default', 8, batch_size, True)

    batch = next(iter(dataloader))
    examples, labels = batch

    attacker = PGDAttack(epsilon=0.15, epsilons=[0.15, 0.10, 0.05, 0.01], alpha=0.15)

    adv_examples = attacker.batch_generate_adv_example(target_model, examples, labels)

    print(examples.shape)
    activation_value, label_predicted = target_model.predict(examples)
    print(activation_value)
    print(label_predicted)

    activation_value, label_predicted = target_model.predict(adv_examples)
    print(activation_value)
    print(label_predicted)

    # for idx, epsilon in enumerate(epsilons):
    #     activation_value, label_predicted = target_model.predict(adv_examples[idx])
    #     print(activation_value)
    #     print(label_predicted)
    #
    # plot_mnist_batch_classes(adv_examples)

# if __name__ == '__main__':
#     target_model = MnistAmplitudeModel()
#     batch_size = 4
#     epsilons = [0.15, 0.1, 0.01, 0.001]
#
#     dataloader = get_dataloader('mnist', 'train', 'default', 8, batch_size, True)
#
#     batch = next(iter(dataloader))
#     examples, labels = batch
#
#     attacker = FGSMAttack(epsilons=epsilons)
#
#     adv_examples = attacker.batch_generate_adv_example(target_model, examples, labels)
#
#     print(examples.shape)
#     activation_value, label_predicted = target_model.predict(examples)
#     print(activation_value)
#     print(label_predicted)
#
#     print(adv_examples.shape)
#     for idx, epsilon in enumerate(epsilons):
#         activation_value, label_predicted = target_model.predict(adv_examples[idx])
#         print(activation_value)
#         print(label_predicted)
#
#     plot_mnist_batch_classes(adv_examples)
