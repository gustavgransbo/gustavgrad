import unittest
import numpy as np
from autograd import Tensor


class TestTensorMul(unittest.TestCase):
    def test_simple_mul(self) -> None:
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)

        t3 = t1 * t2

        assert t3.data.tolist() == [4, 10, 18]

        t3.backward(np.asarray([1.0, 1.0, 1.0]))

        assert t1.grad.tolist() == [4.0, 5.0, 6.0]
        assert t2.grad.tolist() == [1.0, 2.0, 3.0]

    def test_broadcasted_mul1(self) -> None:
        """ In this test t2 is broadcasted by adding a dimension and then
        repeating it's values"""
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        t2 = Tensor([1, 2, 3], requires_grad=True)

        t3 = t1 * t2

        assert t3.data.tolist() == [[1, 4, 9], [4, 10, 18]]

        t3.backward(np.asarray([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))

        assert t1.grad.tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        # The gradient of t2 should be doubled since all it's values are
        # broadcasted to two places.
        assert t2.grad.tolist() == [5.0, 7.0, 9.0]

    def test_broadcasted_mul2(self) -> None:
        """ In this test t2 is broadcasted only by repeating it's values"""
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        t2 = Tensor([[1, 2, 3]], requires_grad=True)

        t3 = t1 * t2

        assert t3.data.tolist() == [[1, 4, 9], [4, 10, 18]]

        t3.backward(np.asarray([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))

        assert t1.grad.tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        # The gradient of t2 should be doubled since all it's values are
        # broadcasted to two places.
        assert t2.grad.tolist() == [[5.0, 7.0, 9.0]]

    def test_broadcasted_mul3(self) -> None:
        """ In this test t2 has shape (3, 1, 2) and is broadcasted across an
        inner dimension"""
        t1 = Tensor(np.ones(shape=(3, 2, 2)), requires_grad=True)
        t2 = Tensor(3 * np.ones(shape=(3, 1, 2)), requires_grad=True)

        t3 = t1 * t2

        np.testing.assert_equal(t3.data, 3 * np.ones(shape=(3, 2, 2)))

        t3.backward(np.ones(shape=(3, 2, 2)))

        np.testing.assert_equal(t1.grad, 3 * np.ones(shape=(3, 2, 2)))
        np.testing.assert_equal(t2.grad, 2 * np.ones(shape=(3, 1, 2)))

    def test_broadcasted_scalar_mul(self) -> None:
        """ In this test t2 is a scalar"""
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        t2 = Tensor(2, requires_grad=True)

        t3 = t1 * t2

        assert t3.data.tolist() == [[2, 4, 6], [8, 10, 12]]

        t3.backward(np.asarray([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))

        assert t1.grad.tolist() == [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
        assert t2.grad.tolist() == 21.0

        # Also try the reverse direction
        t1.zero_grad(), t2.zero_grad()
        t4 = t2 * t1

        assert t4.data.tolist() == [[2, 4, 6], [8, 10, 12]]

        t4.backward(np.asarray([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))

        assert t1.grad.tolist() == [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
        assert t2.grad.tolist() == 21.0

    def test_inplace_mul(self) -> None:
        t1 = Tensor([1, 2, 3], requires_grad=True)
        assert t1.grad.tolist() == [0.0, 0.0, 0.0]
        t2 = Tensor([4, 5, 6], requires_grad=True)
        assert t2.grad.tolist() == [0.0, 0.0, 0.0]

        t1 *= t2
        assert t1.data.tolist() == [4, 10, 18]
        assert t1.grad is None

        # And with a scalar
        t3 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        assert t3.grad.tolist() == [0.0, 0.0, 0.0]
        t3 *= -1

        assert t3.data.tolist() == [-1, -2, -3]
        assert t3.grad is None

        # And with an ndarray
        t4 = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        assert t4.grad.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        t4 *= 2 * np.ones(shape=(2, 3))

        assert t4.data.tolist() == [[2, 4, 6], [8, 10, 12]]
        assert t4.grad is None

    def test_chained_mul(self) -> None:
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)
        t3 = Tensor([7, 8, 9], requires_grad=True)

        t4 = t1 * t2 * t3

        assert t4.data.tolist() == [28, 80, 162]

        t4.backward(np.asarray([1.0, 1.0, 1.0]))

        assert t1.grad.tolist() == [28.0, 40.0, 54.0]
        assert t2.grad.tolist() == [7.0, 16.0, 27.0]
        assert t3.grad.tolist() == [4.0, 10.0, 18.0]
