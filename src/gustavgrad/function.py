""" Activation functions """
import numpy as np

from gustavgrad.tensor import Dependency, Tensor


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """ Stable sigmoid"""
    mask = x > 0
    res = np.zeros_like(x)
    res[mask] = 1 / (1 + np.exp(-x[mask]))
    res[~mask] = np.exp(x[~mask]) / (1 + np.exp(x[~mask]))
    return res


def _sigmoid_prime(x: np.ndarray) -> np.ndarray:
    s = _sigmoid(x)
    return s * (1 - s)


def sigmoid(tensor: Tensor) -> Tensor:
    """Applies the logistic function element-wise to tensor"""
    data = _sigmoid(tensor.data)
    requires_grad = tensor.requires_grad
    depends_on = []
    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * _sigmoid_prime(tensor.data)

        depends_on.append(Dependency(tensor, grad_fn))

    return Tensor(data, requires_grad, depends_on)


def tanh(tensor: Tensor) -> Tensor:
    """Applies hyperbolic tangent element-wise to tensor"""
    data = np.tanh(tensor.data)
    requires_grad = tensor.requires_grad
    depends_on = []
    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - np.tanh(tensor.data) ** 2)

        depends_on.append(Dependency(tensor, grad_fn))

    return Tensor(data, requires_grad, depends_on)
