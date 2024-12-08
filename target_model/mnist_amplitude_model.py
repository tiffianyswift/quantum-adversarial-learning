import torch

from data_preprocess.dataloader import get_dataloader
from model_training_related.ansatz import RealAmplitude
from model_training_related.model import AmplitudeModel
from model_training_related.model_utils import get_predict_label
from model_training_related.observable import TwoClassifyObservable
from target_model.base import BaseTargetModel


class MnistAmplitudeModel(BaseTargetModel):
    def __init__(self):
        n_qubits = 8
        reps = 3
        ansatz = RealAmplitude(n_qubits, reps)
        observable = TwoClassifyObservable(n_qubits)
        target_model = AmplitudeModel(n_qubits, ansatz, observable)
        target_model.load(path="../model_saved/param.npy")
        loss_func = torch.nn.BCELoss()
        super().__init__(target_model, loss_func)

    def predict(self, x):
        activation_value = self.model(x)
        return activation_value, get_predict_label(activation_value, "bce")



