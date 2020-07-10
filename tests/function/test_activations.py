import numpy as np

from gustavgrad import Tensor
from gustavgrad.function import sigmoid, tanh


class TestActivation:
    def test_sigmoid_shape(self) -> None:
        # TODO: Also test that the actual values of result and gradient are
        # correct
        np.random.seed(0)
        t1 = Tensor(np.random.random(size=(3, 3, 3)), requires_grad=True)
        t2 = sigmoid(t1)

        assert t2.shape == (3, 3, 3)

        t2.backward(np.ones(shape=(3, 3, 3)))

    def test_tanh_shape(self) -> None:
        # TODO: Also test that the actual values of result and gradient are
        # correct
        np.random.seed(0)
        t1 = Tensor(np.random.random(size=(3, 3, 3)), requires_grad=True)
        t2 = tanh(t1)

        assert t2.shape == (3, 3, 3)

        t2.backward(np.ones(shape=(3, 3, 3)))
