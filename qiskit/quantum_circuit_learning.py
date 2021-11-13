"""
Basic implementation of the quantum circuit learning algorithm for fitting a function
using quantum supervised learning. This implementation has not been optimized and it
is quite inefficient.

One of the differences between the original paper and this implementation is that the Hamiltonian evolution
circuit block is replace by a sequence of single qubit rotations and multi-qubit entanglement (CNOT)
gates.

Another different with respect to the original paper is that here, for simplicity, we use a gradient-free
optimizer (Nelder-Mead) to avoid computing circuit derivatives using the parametric-shift rule mentioned in
the original paper. Using a gradient-based optimizer would greatly speed up convergence.

The cost operator is built using the QubitOperator representation offered by the Openfermion library.

Original paper: https://arxiv.org/abs/1803.00745
"""

from itertools import combinations, count
from typing import Callable, Tuple, Iterator

import matplotlib.pylab as plt
import numpy as np
from openfermion import QubitOperator, get_sparse_operator
from qiskit import Aer, ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.providers.aer.backends.aerbackend import AerBackend
from scipy.optimize import minimize


np.random.seed(42)

backend: AerBackend = Aer.get_backend("statevector_simulator")

# functions to fit
sin_fn = lambda x: 0.5 * np.cos(-np.pi * x)
x_2 = lambda x: x ** 2


def get_teacher_data(
    fn: Callable, domain: tuple = (-1, 1), n_teacher: int = 100
) -> Tuple[np.ndarray, np.ndarray]:

    start, end = domain
    x_rand = np.sort(np.random.uniform(low=start, high=end, size=n_teacher))
    y_rand = fn(x_rand)

    return x_rand, y_rand


def expectation_value(operator: np.ndarray, circuit: QuantumCircuit):
    job = backend.run(circuit)
    wavefunction = job.result().get_statevector(circuit)
    res = np.vdot(wavefunction, operator.dot(wavefunction))
    return res.real


def cost_operator(n_qubits: int = 2, cost_type="total_mag") -> np.ndarray:
    """Ising cost operator"""
    hamiltonian = QubitOperator()

    for qubit in range(n_qubits):
        hamiltonian += QubitOperator(f"Z{qubit}", coefficient=1.0)

    if cost_type == "zz_ham":
        for qubit in range(n_qubits):
            for qubit2 in range(qubit, n_qubits):
                hamiltonian += QubitOperator(f"Z{qubit} Z{qubit2}", coefficient=1.0)

        if cost_type == "ising":
            for qubit in range(n_qubits):
                hamiltonian += QubitOperator(f"X{qubit}", coefficient=1.0)

    return get_sparse_operator(hamiltonian).toarray()


def quantum_circuit(n_qubits: int = 2, depth: int = 1):
    def feature_map():
        """Fourier feature map"""
        qr = QuantumRegister(n_qubits, "q")
        cr = ClassicalRegister(n_qubits, "c")
        _circuit: QuantumCircuit = QuantumCircuit(qr, cr)

        param = Parameter("phi")
        for i in range(n_qubits):
            _circuit.ry(param, i)

        return _circuit

    def variational_ansatz():
        """Simple hardware efficient ansatz"""
        qr = QuantumRegister(n_qubits, "q")
        cr = ClassicalRegister(n_qubits, "c")
        _circuit: QuantumCircuit = QuantumCircuit(qr, cr)

        comb = list(combinations(list(range(n_qubits)), 2))
        entang_idx = [c for c in comb if abs(c[0] - c[1]) == 1]

        for i in range(depth):
            for j in range(n_qubits):
                param = Parameter(f"theta_{i}{j}1")
                _circuit.rz(param, j)

                param = Parameter(f"theta_{i}{j}2")
                _circuit.rx(param, j)

                if i > 0:
                    param = Parameter(f"theta_{i}{j}3")
                    _circuit.rz(param, j)

            for target in entang_idx:
                _circuit.cnot(target[0], target[1])

        return _circuit

    fm = feature_map()
    ansatz = variational_ansatz()

    return fm + ansatz


def f_eval(x: float, params: np.ndarray, operator: np.ndarray, circuit: QuantumCircuit = None):
    new_parameters = [x] + list(params)
    new_circ = circuit.assign_parameters(new_parameters)
    res = expectation_value(operator, new_circ)
    return res


def loss(
    params: np.ndarray,
    x: np.ndarray = None,
    y_teacher: np.ndarray = None,
    circuit: QuantumCircuit = None,
    operator: np.ndarray = None,
):

    if x is None or y_teacher is None or circuit is None:
        raise ValueError("You must provide the input coordinates data")

    y_pred = np.array([f_eval(point, params, operator, circuit=circuit) for point in x])
    res = np.sqrt(np.sum((y_pred - y_teacher) ** 2))
    return res


if __name__ == "__main__":
    from functools import partial

    x_teacher, y_teacher = get_teacher_data(sin_fn, n_teacher=50)

    n_qubits = 6
    circuit = quantum_circuit(n_qubits=n_qubits, depth=2)
    operator = cost_operator(n_qubits=n_qubits)

    print("Quantum circuit")
    print(circuit)
    print()

    n_parameters = circuit.num_parameters - 1
    initial_params = np.random.rand(n_parameters) * 10.0

    objective_fn = partial(
        loss, x=x_teacher, y_teacher=y_teacher, circuit=circuit, operator=operator
    )
    res = objective_fn(initial_params)
    print(f"Initial loss: {res}")

    epoch_counter = count()

    def minimize_cb(xk: np.ndarray, counter: Iterator = epoch_counter):
        n_epoch = next(epoch_counter)
        print(f"Epoch #{n_epoch} - Loss: {objective_fn(xk)}")

    result = minimize(
        objective_fn,
        x0=initial_params,
        method="Nelder-Mead",
        tol=1e-3,
        options={"maxiter": 500, "disp": True},
        callback=minimize_cb,
    )

    y_initial = [
        f_eval(element, initial_params, operator, circuit=circuit) for element in x_teacher
    ]
    y_optimal = [f_eval(element, result.x, operator, circuit=circuit) for element in x_teacher]

    plt.figure()
    plt.scatter(x_teacher, y_teacher, label="Teacher", marker="o")
    plt.plot(x_teacher, y_initial, label="Initial prediction", color="green", alpha=0.7)
    plt.plot(x_teacher, y_optimal, label="Final prediction")
    plt.legend()
    plt.show()
