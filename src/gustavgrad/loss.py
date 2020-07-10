""" Loss functions"""
from abc import ABC, abstractmethod

import numpy as np

from gustavgrad.function import _sigmoid
from gustavgrad.tensor import Dependency, Tensor, Tensorable, ensure_tensor


class Loss(ABC):
    @abstractmethod
    def loss(self, target: Tensorable, pred: Tensor) -> Tensor:
        raise NotImplementedError


class SquaredErrorLoss(Loss):
    """ Squared error loss """

    def loss(self, target: Tensorable, pred: Tensor) -> Tensor:
        error = ensure_tensor(target) - pred
        return (error * error).sum()


class LogitBinaryCrossEntropy(Loss):
    """ Binary cross entropy with logits
    Applies the logistic function (sigmoid) to logits before calculating loss
    """

    def loss(self, target: Tensorable, logits: Tensor) -> Tensor:

        tensor_target = ensure_tensor(target)

        # Calculate stable binary cross entropy loss
        x = np.clip(logits.data, 0, None)
        neg_abs = -np.abs(logits.data)
        data = (
            x - logits.data * tensor_target.data + np.log(1 + np.exp(neg_abs))
        )
        # Just do the mean for now
        data = data.mean()

        requires_grad = logits.requires_grad
        depends_on = []
        if requires_grad:

            def grad_fn(grad: np.ndarray) -> np.ndarray:
                return grad * (_sigmoid(logits.data) - tensor_target.data)

            depends_on.append(Dependency(logits, grad_fn))

        return Tensor(data, requires_grad, depends_on)
