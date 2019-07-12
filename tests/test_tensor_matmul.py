import unittest
import pytest
import numpy as np
from autograd import Tensor

class TestTensorMatMul(unittest.TestCase):
    def test_simple_matmul(self) -> None:
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        t2 = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)

        t3 = t1 @ t2

        assert t3.data.tolist() == [[22, 28], [49, 64]]

        t3.backward(np.asarray([[1., 2.], [3., 4.]]))

        assert t1.grad.tolist() == [[(1.+4.), (3.+8.), (5.+12.)], [(3.+8.), (9.+16.), (15.+24.)]]
        assert t2.grad.tolist() == [[(1.+12.), (2.+16.)], [(2.+15.), (4.+20.)], [(3.+18.), (6.+24.)]]

    def test_chained_mul_shape(self) -> None:
        t1 = Tensor(np.ones(shape=(2, 4)), requires_grad=True)
        t2 = Tensor(np.ones(shape=(4, 3)), requires_grad=True)
        t3 = Tensor(np.ones(shape=(3, 1)), requires_grad=True)

        t4 = t1 @ t2 @ t3

        assert t4.shape == (2, 1)

        # TODO: Extend to also test values and gradient