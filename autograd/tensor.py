"""
Tensor Class and affiliated functions
"""
from typing import List, NamedTuple, Callable, Optional, Union
import numpy as np

class Dependency(NamedTuple):
    """
    A dependency consists of a tensor and a gradient function.
    The idea is that if tensor t2 = f(t1), then t2 should have the dependency:
    (t1, df/dt1*dt2/dtf).
    If we have a gradient dJ/dt2, we can now calculate the gradient of t1 as:
    dJ/dt1 = df/dt1*dt2/dtf*dJ/dt2
    """
    tensor : 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]

# Types that can be converted to an ndarray
Arrayable = Union[np.ndarray, list, float]
def ensure_array(data: Arrayable) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    else:
        return np.asarray(data)

# Types that can be converted to a tensor
Tensorable = Union['Tensor', np.ndarray, float]
def ensure_tensor(data: Tensorable) -> 'Tensor':
    if isinstance(data, Tensor):
        return data
    else:
        return Tensor(data)

class Tensor:
    """
    A tensor is an ndarray that keeps track of it's own gradient,
    and can propagate it's gradient backward to tensors it depends on.

    For example, if t3 = t1 + t2, then the tensor t3 depends on t1 and t2.
    """

    def __init__(self, 
                data: Arrayable, 
                requires_grad: bool = False, 
                depends_on: List[Dependency] = []) -> None:
        
        # Private, see self.data() property for explanation TODO: Perhaps cast to float?
        self._data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on
        self.grad : Optional['Tensor'] = None
        self.shape: tuple = self._data.shape

        if self.requires_grad:
            self.zero_grad()

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value: np.ndarray) -> None:
        """ Invalidate the gradient of the tensor if data is modified in-place"""
        self.grad = None
        self._data = value

    def zero_grad(self) -> None:
        """ Set gradient to 0"""
        self.grad = np.zeros_like(self.data, dtype=np.float)

    def backward(self, grad: np.ndarray = None) -> None:
        """ Propagate a gradient backward to all dependencies"""

        if grad is None:
            if self.shape == ():
                grad = np.asarray(1.)
            else:
                raise RuntimeError("Must specify gradient for non-zero Tensor")

        self.grad += grad

        for tensor, grad_fn in self.depends_on:
            tensor.backward(grad_fn(grad))

    def __add__(self, other: 'Tensor') -> 'Tensor':
        return _add(self, ensure_tensor(other))

    def __radd__(self, other: 'Tensor') -> 'Tensor':
        return _add(ensure_tensor(other), self)

    def __iadd__(self, other:'Tensor') -> 'Tensor':
        self.data += ensure_tensor(other).data
        return self

    def sum(self, axis=None) -> 'Tensor':
        return _sum(self, axis)


""" Functions on tensors """
def _add(t1: Tensor, t2: Tensor) -> Tensor:
    """ Adds two tensors

    Gradients of t1 and t2 are the same as t3, so the gradient function is the identity function,
    UNLESS some dimension of t1 or t2 was broadcasted in the addition, in which case the gradient
    is increased.
    """

    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            return sum_out_broadcasted_dims(grad, t1.shape)

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return sum_out_broadcasted_dims(grad, t2.shape)
        depends_on.append(Dependency(t2, grad_fn2))
    
    return Tensor(data, requires_grad, depends_on)

def _sum(tensor: Tensor, axis=None) -> Tensor:
    """
    Sum a tensor along given axis. The gradient function is simply the identity function.
    """
    data = tensor.data.sum(axis)
    requires_grad = tensor.requires_grad
    depends_on = []
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            if axis is not None:
                # Handle broadcasting correctly
                grad = np.expand_dims(grad, axis)
            return grad * np.ones_like(tensor.data)
        depends_on.append(Dependency(tensor, grad_fn))
    
    return Tensor(data, requires_grad, depends_on)
    

""" Helper Functions """
def sum_out_broadcasted_dims(grad: np.ndarray, tensor_shape: tuple) -> np.ndarray:
    """ Handle Broadcasting:
    1. If t1 is shape(N,) and grad is shape (M, K, N), then the grad of t3 with respect to t1
    should be summed across the added dimensions.
    2. If t1 has any dimensions with a single entry, eg. (N, 1, K), then that dimension will also
    have been broadcasted if the same deimension is >1 for t2. Thus, the gradient needs to be summed
    across this dimension as well.
    """
    # Sum across added dimensions (Step 1)
    added_dims = grad.ndim - len(tensor_shape)
    for _ in range(added_dims):
        grad = grad.sum(0)
    # Sum across singular dimensions (Step 2)
    for i, dim in enumerate(tensor_shape):
        if dim == 1:
            grad = grad.sum(i, keepdims=True)
    return grad