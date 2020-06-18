import unittest
import pytest
import numpy as np
from autograd import Tensor


class TestTensorSlice(unittest.TestCase):
    def test_single_slice(self) -> None:
        t1 = Tensor(np.arange(9).reshape(3, 3), requires_grad=True)

        t2 = t1[:2]
        assert t2.data.tolist() == [[0, 1, 2], [3, 4, 5]]

        t2.backward(np.ones(shape=(2, 3)))

        assert t1.grad.tolist() == [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ]

    def test_double_slice(self) -> None:
        t1 = Tensor(np.arange(9).reshape(3, 3), requires_grad=True)

        t2 = t1[:2, :2]
        assert t2.data.tolist() == [[0, 1], [3, 4]]

        t2.backward(np.ones(shape=(2, 2)))

        assert t1.grad.tolist() == [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]

    def test_boolean_slice(self) -> None:
        t1 = Tensor(np.arange(9).reshape(3, 3), requires_grad=True)

        t2 = t1[t1.data >= 5]
        assert t2.data.tolist() == [5, 6, 7, 8]

        t2.backward(np.ones(shape=(4,)))

        assert t1.grad.tolist() == [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
