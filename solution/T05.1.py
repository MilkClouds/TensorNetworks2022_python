import torch, pennylane as qml
from pennylane import numpy as np

X = qml.PauliX(wires=0).matrix()
Y = qml.PauliY(wires=0).matrix()
Z = qml.PauliZ(wires=0).matrix()
I = qml.Identity(wires=0).matrix()
c = np.array(((0, 1,), (0, 0)))

N = 5
t = 2 ** (np.arange(N) / 2)
# H = 1

# for l in range(1, N):
    # H += -t[l] * np.kron(np.kron(np.kron(np.eye(2 ** (l - 1)), c), np.eye(2 ** (N - l - 1))), c)
    # H += -t[l]