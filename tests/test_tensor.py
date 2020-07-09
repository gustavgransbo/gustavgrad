import unittest

import numpy as np
import pytest

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


class TestTensorBasics(unittest.TestCase):
    def test_init_scalar_data(self) -> None:
        tensor = Tensor(1)

        assert type(tensor.data) is np.ndarray
        assert tensor.shape == ()

    def test_init_list_data(self) -> None:
        tensor = Tensor([1, 2, 3])

        assert type(tensor.data) is np.ndarray
        assert tensor.shape == (3,)

    def test_init_ndarray_data(self) -> None:
        tensor = Tensor(np.ones((3, 3, 3)))

        assert type(tensor.data) is np.ndarray
        assert tensor.shape == (3, 3, 3)

    def test_reassign_data(self) -> None:
        tensor = Tensor([1, 2, 3])

        tensor.data = np.array([3, 2, 1])
        assert tensor.data.tolist() == [3, 2, 1]

    def test_reassign_data_wrong_shape(self) -> None:
        tensor = Tensor([1, 2, 3])

        with pytest.raises(RuntimeError):
            tensor.data = np.array([3])

    def test_backward_without_grad(self) -> None:
        tensor = Tensor([1], requires_grad=True)
        with pytest.raises(RuntimeError):
            tensor.backward()

    def test_backward_without_grad_zero_tensor(self) -> None:
        tensor = Tensor(1, requires_grad=True)
        tensor.backward()
        assert tensor.grad.tolist() == 1
