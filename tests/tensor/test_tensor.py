import numpy as np
import pytest

from gustavgrad import Tensor


class TestTensorRepr:
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


class TestTensorBasics:
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


class TestTensorNoGrad:
    @pytest.fixture
    def tensor_with_grad(self) -> Tensor:
        t1 = Tensor(1, requires_grad=True)
        t2 = t1 + 1
        t2.backward()

        assert not np.allclose(t1.grad, 0)

        return t1

    def test_requires_grad_false_in_context_manager(self) -> None:
        tensor = Tensor(1, requires_grad=True)
        with tensor.no_grad():
            assert not tensor.requires_grad

    def test_requires_grad_resets(self) -> None:
        tensor = Tensor(1, requires_grad=True)
        with tensor.no_grad():
            pass
        assert tensor.requires_grad

    def test_requires_grad_resets_after_exception(self) -> None:
        tensor = Tensor(1, requires_grad=True)
        try:
            with tensor.no_grad():
                raise Exception()
        except Exception:
            pass
        assert tensor.requires_grad

    def test_dependencies(self) -> None:
        t1 = Tensor(1, requires_grad=True)
        with t1.no_grad():
            t2 = t1 + 1

        assert not t2.requires_grad
        assert t1 not in [dependency.tensor for dependency in t2.depends_on]

    def test_grad_intact(self, tensor_with_grad) -> None:
        initial_grad = tensor_with_grad.grad
        with tensor_with_grad.no_grad():
            _ = tensor_with_grad * 100

        assert tensor_with_grad.grad == initial_grad
