import unittest
import pytest
import numpy as np
from autograd import Tensor
from autograd.module import Module, Parameter

class TestModule(unittest.TestCase):

    class Model(Module):
        def __init__(self) -> None:
            self.layer1 = Parameter(2, 10)
            self.layer2 = Parameter(10, 5)

        def predict(self, x: Tensor) -> Tensor:
            # x has shape(batch_size, 2)
            x = x @ self.layer1    # (batch_size, 10)
            x = x @ self.layer2    # (batch_size, 5)
            return x
    
    def test_zero_grad(self) -> None:
        model = self.Model()

        y = model.predict(Tensor([[1, 2], [1, 3]]))

        y.backward(np.ones(shape=(2, 5)))

        assert not np.allclose(model.layer1.grad, np.zeros_like(model.layer1.grad))
        assert not np.allclose(model.layer2.grad, np.zeros_like(model.layer2.grad))

        model.zero_grad()

        assert np.allclose(model.layer1.grad, np.zeros_like(model.layer1.grad))
        assert np.allclose(model.layer2.grad, np.zeros_like(model.layer2.grad))
