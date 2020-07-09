import unittest

import numpy as np

from gustavgrad import Tensor


class TestTensorRepr(unittest.TestCase):
    def test_1d_with_grad(self) -> None:
        tensor = Tensor([1, 2, 3], requires_grad=True)

        assert repr(tensor) == "Tensor(data=[1 2 3], requires_grad=True)"

    def test_1d_without_grad(self) -> None:
        tensor = Tensor([1, 2, 3], requires_grad=False)

        assert repr(tensor) == "Tensor(data=[1 2 3], requires_grad=False)"

    def test_2d_with_grad(self) -> None:
        tensor = Tensor(np.ones((2, 2)), requires_grad=True)

        assert (
            repr(tensor) == "Tensor(data=\n"
            "[[1. 1.]\n"
            " [1. 1.]], requires_grad=True)"
        )
