# import numpy as np
import pennylane as qml
from pennylane import numpy as np

X = qml.PauliX(wires=0).matrix()
Y = qml.PauliY(wires=0).matrix()
Z = qml.PauliZ(wires=0).matrix()
I = qml.Identity(wires=0).matrix()

H = 0
for i, j in ((0, 1), (1, 2), (2, 0)):
    S = ["I", "I", "I"]
    for s in "XYZ":
        S[i] = s; S[j] = s
        H += qml.pauli.string_to_pauli_word(''.join(S))
        S = ["I", "I", "I"]

H = qml.matrix(H) / 4
print(np.linalg.eigh(H)[:1])

N = 11
H = np.diag(-np.exp(1j * np.arange(1, N)), -1)
H += np.matrix.getH(H)

def ground_state(H):
    w, v = np.linalg.eigh(H)
    w[abs(w) < 1e-9] = 0
    return np.sum(w[w < 0]), 2 ** np.sum(w == 0), w

print(ground_state(H)[:2])

c = np.array([[0, 1], [0, 0]])
