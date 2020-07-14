import numpy as np

from gustavgrad import Tensor
from gustavgrad.function import sigmoid, tanh


class TestActivation:
    def test_sigmoid(self) -> None:
        t1 = Tensor(np.zeros(shape=(3, 3, 3)), requires_grad=True)
        t2 = sigmoid(t1)

        assert t2.shape == (3, 3, 3)
        np.testing.assert_allclose(t2.data, 0.5)

    def test_sigmoid_grad(self) -> None:
        t1 = Tensor(np.zeros(shape=(3, 3, 3)), requires_grad=True)
        t2 = sigmoid(t1)
        t2.backward(1)

        np.testing.assert_allclose(t1.grad, 0.25)

    def test_sigmoid_no_grad(self) -> None:
        t1 = Tensor(np.zeros(shape=(3, 3, 3)), requires_grad=False)
        t2 = sigmoid(t1)

        assert t2.shape == (3, 3, 3)
        assert not t2.requires_grad

    def test_tanh(self) -> None:
        np.random.seed(0)
        t1 = Tensor(np.ones(shape=(3, 3, 3)) * 1_000, requires_grad=True)
        t2 = tanh(t1)

        assert t2.shape == (3, 3, 3)
        np.testing.assert_allclose(t2.data, 1)

    def test_tanh_grad(self) -> None:
        np.random.seed(0)
        t1 = Tensor(np.ones(shape=(3, 3, 3)) * 1_000, requires_grad=True)
        t2 = tanh(t1)
        t2.backward(1)

        np.testing.assert_allclose(t1.grad, 0)

    def test_tanh_no_grad(self) -> None:
        t1 = Tensor(np.zeros(shape=(3, 3, 3)), requires_grad=False)
        t2 = tanh(t1)

        assert t2.shape == (3, 3, 3)
        assert not t2.requires_grad
