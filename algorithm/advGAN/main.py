from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from Utils.Plot import PlotLoss

if __name__ == '__main__':
    PlotLoss("metric/val", window_size=10, stride=5)

