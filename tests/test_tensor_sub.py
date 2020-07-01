import unittest

import numpy as np

from gustavgrad import Tensor


class TestTensorSub(unittest.TestCase):
    def test_simple_sub(self) -> None:
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)

        t3 = t1 - t2

        assert t3.data.tolist() == [-3, -3, -3]

        t3.backward(np.asarray([1.0, 1.0, 1.0]))

        assert t1.grad.tolist() == [1.0, 1.0, 1.0]
        assert t2.grad.tolist() == [-1.0, -1.0, -1.0]

    def test_broadcasted_sub1(self) -> None:
        """ In this test t2 is broadcasted by adding a dimension and then
        repeating it's values"""
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        t2 = Tensor([1, 2, 3], requires_grad=True)

        t3 = t1 - t2

        assert t3.data.tolist() == [[0, 0, 0], [3, 3, 3]]

        t3.backward(np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

        assert t1.grad.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        assert t2.grad.tolist() == [-5.0, -7.0, -9.0]

    def test_broadcasted_sub2(self) -> None:
        """ In this test t2 is broadcasted only by repeating it's values"""
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        t2 = Tensor([[1, 2, 3]], requires_grad=True)

        t3 = t1 - t2

        assert t3.data.tolist() == [[0, 0, 0], [3, 3, 3]]

        t3.backward(np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

        assert t1.grad.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        assert t2.grad.tolist() == [[-5.0, -7.0, -9.0]]

    def test_broadcasted_sub3(self) -> None:
        """ In this test t2 has shape (3, 1, 2) and is broadcasted across an
        inner dimension"""
        t1 = Tensor(np.ones(shape=(3, 2, 2)), requires_grad=True)
        t2 = Tensor(np.ones(shape=(3, 1, 2)), requires_grad=True)

        t3 = t1 - t2

        np.testing.assert_equal(t3.data, np.zeros(shape=(3, 2, 2)))

        t3.backward(np.ones(shape=(3, 2, 2)))

        np.testing.assert_equal(t1.grad, np.ones(shape=(3, 2, 2)))
        # The gradient of t2 should be doubled since all it's values are
        # broadcasted to two places.
        np.testing.assert_equal(t2.grad, -2 * np.ones(shape=(3, 1, 2)))

    def test_broadcasted_scalar_sub(self) -> None:
        """ In this test t2 is a scalar"""
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        t2 = Tensor(1, requires_grad=True)

        t3 = t1 - t2

        assert t3.data.tolist() == [[0, 1, 2], [3, 4, 5]]

        t3.backward(np.asarray([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))

        assert t1.grad.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        # The gradient of t2 should be doubled since all it's values are
        # broadcasted to two places.
        assert t2.grad.tolist() == -6.0

        # Also try the reverse direction
        t1.zero_grad(), t2.zero_grad()
        t4 = t2 - t1

        assert t4.data.tolist() == [[0, -1, -2], [-3, -4, -5]]

        t4.backward(np.asarray([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))

        assert t1.grad.tolist() == [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]
        # The gradient of t2 should be doubled since all it's values are
        # broadcasted to two places.
        assert t2.grad.tolist() == 6.0

    def test_inplace_sub(self) -> None:
        t1 = Tensor([1, 2, 3], requires_grad=True)
        assert t1.grad.tolist() == [0.0, 0.0, 0.0]
        t2 = Tensor([4, 5, 6], requires_grad=True)
        assert t2.grad.tolist() == [0.0, 0.0, 0.0]

        t1 -= t2
        assert t1.data.tolist() == [-3, -3, -3]
        assert t1.grad is None

        # And with a scalar
        t3 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        assert t3.grad.tolist() == [0.0, 0.0, 0.0]
        t3 -= 1

        assert t3.data.tolist() == [0.0, 1.0, 2.0]
        assert t3.grad is None

        # And with an ndarray
        t4 = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        assert t4.grad.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        t4 -= np.ones(shape=(2, 3))

        assert t4.data.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        assert t4.grad is None

    def test_chained_sub(self) -> None:
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)
        t3 = Tensor([7, 8, 9], requires_grad=True)

        t4 = t3 - t2 - t1

        assert t4.data.tolist() == [2, 1, 0]

        t4.backward(np.asarray([1.0, 1.0, 1.0]))

        assert t1.grad.tolist() == [-1.0, -1.0, -1.0]
        assert t2.grad.tolist() == [-1.0, -1.0, -1.0]
        assert t3.grad.tolist() == [1.0, 1.0, 1.0]
