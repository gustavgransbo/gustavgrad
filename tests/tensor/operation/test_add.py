import numpy as np

from gustavgrad import Tensor


class TestTensorAdd:
    def test_simple_add_result(self) -> None:
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)
        t3 = t1 + t2

        assert t3.data.tolist() == [5, 7, 9]
        assert t3.requires_grad

    def test_simple_add_backward(self) -> None:
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)
        t3 = t1 + t2
        t3.backward(np.asarray([1.0, 1.0, 1.0]))

        assert t1.grad.tolist() == [1.0, 1.0, 1.0]
        assert t2.grad.tolist() == [1.0, 1.0, 1.0]

    def test_add_depends_on_requires_grad_true(self) -> None:
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)
        t3 = t1 + t2

        assert t1 in [dependency.tensor for dependency in t3.depends_on]
        assert t2 in [dependency.tensor for dependency in t3.depends_on]

    def test_add_depends_on_requires_grad_false(self) -> None:
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=False)
        t3 = t1 + t2

        assert t1 in [dependency.tensor for dependency in t3.depends_on]
        assert t2 not in [dependency.tensor for dependency in t3.depends_on]

    def test_simple_add_no_grad(self) -> None:
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([4, 5, 6])

        t3 = t1 + t2

        assert t3.data.tolist() == [5, 7, 9]
        assert not t3.requires_grad

    def test_scalar_add_result(self) -> None:
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = t1 + 1
        assert t2.data.tolist() == [2, 3, 4]

    def test_scalar_add_backward(self) -> None:
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = t1 + 1
        t2.backward(np.asarray([1.0, 1.0, 1.0]))
        assert t1.grad.tolist() == [1.0, 1.0, 1.0]

    def test_array_add_result(self) -> None:
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = t1 + np.array([4, 5, 6])
        assert t2.data.tolist() == [5, 7, 9]

    def test_array_add_backward(self) -> None:
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = t1 + np.array([4, 5, 6])
        t2.backward(np.asarray([1.0, 1.0, 1.0]))
        assert t1.grad.tolist() == [1.0, 1.0, 1.0]

    def test_scalar_radd_result(self) -> None:
        t1 = Tensor([1, 2, 3], requires_grad=True)

        t2 = 1 + t1
        assert type(t2) is Tensor
        assert t2.data.tolist() == [2, 3, 4]

    def test_scalar_radd_backward(self) -> None:
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = 1 + t1
        t2.backward(np.asarray([1.0, 1.0, 1.0]))
        assert t1.grad.tolist() == [1.0, 1.0, 1.0]

    def test_broadcasted_add_extra_dimension(self) -> None:
        """ In this test t2 is broadcasted by adding a dimension and then
        repeating it's values"""
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        t2 = Tensor([1, 2, 3], requires_grad=True)

        t3 = t1 + t2

        assert t3.data.tolist() == [[2, 4, 6], [5, 7, 9]]

        t3.backward(np.asarray([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))

        assert t1.grad.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        # The gradient of t2 should be doubled since all it's values are
        # broadcasted to two places.
        assert t2.grad.tolist() == [2.0, 2.0, 2.0]

    def test_broadcasted_add_same_dimension(self) -> None:
        """ In this test t2 is broadcasted only by repeating it's values"""
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        t2 = Tensor([[1, 2, 3]], requires_grad=True)

        t3 = t1 + t2

        assert t3.data.tolist() == [[2, 4, 6], [5, 7, 9]]

        t3.backward(np.asarray([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))

        assert t1.grad.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        # The gradient of t2 should be doubled since all it's values are
        # broadcasted to two places.
        assert t2.grad.tolist() == [[2.0, 2.0, 2.0]]

    def test_broadcasted_add_inner_dimension(self) -> None:
        """ In this test t2 has shape (3, 1, 2) and is broadcasted across an
        inner dimension"""
        t1 = Tensor(np.ones(shape=(3, 2, 2)), requires_grad=True)
        t2 = Tensor(np.ones(shape=(3, 1, 2)), requires_grad=True)

        t3 = t1 + t2

        np.testing.assert_equal(t3.data, 2 * np.ones(shape=(3, 2, 2)))

        t3.backward(np.ones(shape=(3, 2, 2)))

        np.testing.assert_equal(t1.grad, np.ones(shape=(3, 2, 2)))
        # The gradient of t2 should be doubled since all it's values are
        # broadcasted to two places.
        np.testing.assert_equal(t2.grad, 2 * np.ones(shape=(3, 1, 2)))

    def test_broadcasted_scalar_add(self) -> None:
        """ In this test t2 is a scalar"""
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        t2 = Tensor(1, requires_grad=True)

        t3 = t1 + t2

        assert t3.data.tolist() == [[2, 3, 4], [5, 6, 7]]

        t3.backward(np.asarray([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))

        assert t1.grad.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        # The gradient of t2 should be doubled since all it's values are
        # broadcasted to two places.
        assert t2.grad.tolist() == 6.0

        # Also try the reverse direction
        t1.zero_grad(), t2.zero_grad()
        t4 = t2 + t1

        assert t4.data.tolist() == [[2, 3, 4], [5, 6, 7]]

        t4.backward(np.asarray([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))

        assert t1.grad.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        # The gradient of t2 should be doubled since all it's values are
        # broadcasted to two places.
        assert t2.grad.tolist() == 6.0

    def test_inplace_add(self) -> None:
        t1 = Tensor([1, 2, 3], requires_grad=True)
        assert t1.grad.tolist() == [0.0, 0.0, 0.0]
        t2 = Tensor([4, 5, 6], requires_grad=True)
        assert t2.grad.tolist() == [0.0, 0.0, 0.0]

        t1 += t2
        assert t1.data.tolist() == [5, 7, 9]
        assert t1.grad is None

        # And with a scalar
        t3 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        assert t3.grad.tolist() == [0.0, 0.0, 0.0]
        t3 += 1

        assert t3.data.tolist() == [2, 3, 4]
        assert t3.grad is None

        # And with an ndarray
        t4 = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        assert t4.grad.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        t4 += np.ones(shape=(2, 3))

        assert t4.data.tolist() == [[2, 3, 4], [5, 6, 7]]
        assert t4.grad is None

    def test_chained_add(self) -> None:
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)
        t3 = Tensor([7, 8, 9], requires_grad=True)

        t4 = t1 + t2 + t3

        assert t4.data.tolist() == [12, 15, 18]

        t4.backward(np.asarray([1.0, 1.0, 1.0]))

        assert t1.grad.tolist() == [1.0, 1.0, 1.0]
        assert t2.grad.tolist() == [1.0, 1.0, 1.0]
        assert t3.grad.tolist() == [1.0, 1.0, 1.0]
