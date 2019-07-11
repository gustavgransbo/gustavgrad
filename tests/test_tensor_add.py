import unittest
import pytest
import numpy as np
from autograd import Tensor
from autograd.tensor import _add

class TestTensorAdd(unittest.TestCase):
    def test_simple_add(self) -> None:
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)

        t3 = _add(t1, t2)

        assert t3.data.tolist() == [5, 7, 9]

        t3.backward(np.asarray([1., 1., 1.]))

        assert t1.grad.tolist() == [1., 1., 1.]
        assert t2.grad.tolist() == [1., 1., 1.]

    def test_broadcasted_add1(self) -> None:
        """ In this test t2 is broadcasted by adding a dimension and then repeating it's values"""
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        t2 = Tensor([1, 2, 3], requires_grad=True)

        t3 = _add(t1, t2)

        assert t3.data.tolist() == [[2, 4, 6], [5, 7, 9]]

        t3.backward(np.asarray([[1., 1., 1.], [1., 1., 1.]]))

        assert t1.grad.tolist() == [[1., 1., 1.], [1., 1., 1.]]
        # The gradient of t2 should be doubled since all it's values are broadcasted to two places.
        assert t2.grad.tolist() == [2., 2., 2.]

    def test_broadcasted_add2(self) -> None:
        """ In this test t2 is broadcasted only by repeating it's values"""
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        t2 = Tensor([[1, 2, 3]], requires_grad=True)

        t3 = _add(t1, t2)

        assert t3.data.tolist() == [[2, 4, 6], [5, 7, 9]]

        t3.backward(np.asarray([[1., 1., 1.], [1., 1., 1.]]))

        assert t1.grad.tolist() == [[1., 1., 1.], [1., 1., 1.]]
        # The gradient of t2 should be doubled since all it's values are broadcasted to two places.
        assert t2.grad.tolist() == [[2., 2., 2.]]
    
    def test_broadcasted_add3(self) -> None:
        """ In this test t2 has shape (3, 1, 2) and is broadcasted across an inner dimension"""
        t1 = Tensor(np.ones(shape=(3,2,2)), requires_grad=True)
        t2 = Tensor(np.ones(shape=(3,1,2)), requires_grad=True)

        t3 = _add(t1, t2)

        np.testing.assert_equal(t3.data, 2 * np.ones(shape=(3,2,2)))

        t3.backward(np.ones(shape=(3,2,2)))

        np.testing.assert_equal(t1.grad, np.ones(shape=(3,2,2)))
        # The gradient of t2 should be doubled since all it's values are broadcasted to two places.
        np.testing.assert_equal(t2.grad, 2 * np.ones(shape=(3,1,2)))

    def test_broadcasted_scalar_add(self) -> None:
        """ In this test t2 is a scalar"""
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        t2 = Tensor(1, requires_grad=True)

        t3 = _add(t1, t2)

        assert t3.data.tolist() == [[2, 3, 4], [5, 6, 7]]

        t3.backward(np.asarray([[1., 1., 1.], [1., 1., 1.]]))

        assert t1.grad.tolist() == [[1., 1., 1.], [1., 1., 1.]]
        # The gradient of t2 should be doubled since all it's values are broadcasted to two places.
        assert t2.grad.tolist() == 6.

        # Also try the reverse direction
        t1.zero_grad(), t2.zero_grad()
        t4 = _add(t2, t1)

        assert t4.data.tolist() == [[2, 3, 4], [5, 6, 7]]

        t4.backward(np.asarray([[1., 1., 1.], [1., 1., 1.]]))

        assert t1.grad.tolist() == [[1., 1., 1.], [1., 1., 1.]]
        # The gradient of t2 should be doubled since all it's values are broadcasted to two places.
        assert t2.grad.tolist() == 6.

if __name__ == "__main__":
    """For debugging"""
    TestTensorAdd().test_simple_add()