import unittest
import pytest
import numpy as np
from autograd import Tensor


class TestTensorSum(unittest.TestCase):
    def test_simple_sum(self) -> None:
        t1 = Tensor([1, 2, 3], requires_grad=True)

        t2 = t1.sum()
        assert t2.data.tolist() == 6

        t2.backward()
        assert t1.grad.tolist() == [1.0, 1.0, 1.0]

        # Also try with a specified grad
        t1.zero_grad()
        t3 = t1.sum()
        assert t3.data.tolist() == 6

        t2.backward(2)
        assert t1.grad.tolist() == [2.0, 2.0, 2.0]

    def test_axis_sum(self) -> None:
        t1 = Tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], requires_grad=True)

        # First axis
        t2 = t1.sum(0)
        assert t2.data.tolist() == [2, 4, 6]

        t2.backward(np.asarray([1.0, 2.0, 3.0]))
        assert t1.grad.tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]

        # Second axis
        t1.zero_grad()
        t3 = t1.sum(1)
        assert t3.data.tolist() == [6, 6]

        t3.backward(np.asarray([1.0, 2.0]))
        assert t1.grad.tolist() == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
