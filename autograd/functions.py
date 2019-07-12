""" Activation functions """
from autograd.tensor import Tensor, Dependency
import numpy as np

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def _sigmoid_inverse(x: np.ndarray) -> np.ndarray:
    s = _sigmoid(x)
    return s * (1 - s)

def sigmoid(tensor: Tensor) -> Tensor:
    data = _sigmoid(tensor.data)
    requires_grad = tensor.requires_grad
    depends_on = []
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * _sigmoid_inverse(tensor.data)
        depends_on.append(Dependency(tensor, grad_fn))

    return Tensor(data, requires_grad, depends_on)

def tanh(tensor: Tensor) -> Tensor:
    data = np.tanh(tensor.data)
    requires_grad = tensor.requires_grad
    depends_on = []
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - np.tanh(tensor.data) ** 2)
        depends_on.append(Dependency(tensor, grad_fn))
    
    return Tensor(data, requires_grad, depends_on)