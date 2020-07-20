# gustavgrad
[![Tests](https://github.com/gustavgransbo/gustavgrad/workflows/Tests/badge.svg)](https://github.com/gustavgransbo/gustavgrad/actions?workflow=Tests)
[![codecov](https://codecov.io/gh/gustavgransbo/gustavgrad/branch/master/graph/badge.svg)](https://codecov.io/gh/gustavgransbo/gustavgrad)
[![PyPI](https://img.shields.io/pypi/v/gustavgrad.svg)](https://pypi.org/project/gustavgrad)
[![Python Version](https://img.shields.io/pypi/pyversions/gustavgrad.svg)](https://github.com/gustavgransbo/gustavgrad)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An autograd library built on NumPy, inspired by [Joel Grus's livecoding](https://github.com/joelgrus/autograd/tree/master).

## Installation
### With pip
```bash
pip install gustavgrad
```

## How to use the library
### `Tensor` operations
The `Tensor` class is the cornerstone of `gustavgrad`.
It behaves much like an ordinary `numpy.ndarray`.

`Tensors` can be added together,
```python
>>> from gustavgrad import Tensor
>>> x = Tensor([1, 2, 3])
>>> x + x
Tensor(data=[2 4 6], requires_grad=False)
```
... subracted from each other,
```python
>>> x - x
Tensor(data=[0 0 0], requires_grad=False)
```
... their dot-product can calculated,
```python
>>> x * x
Tensor(data=[1 4 9], requires_grad=False)
```
... and they can be multiplied with each other.
```python
>>> y = Tensor([[1], [2], [3]])
>>> x @ y
Tensor(data=[14], requires_grad=False)
```
`Tensor` operations also support broadcasting:
```python
>>> x * 3
Tensor(data=[3 6 9], requires_grad=False)
>>> z = Tensor([[1, 2, 3], [4, 5, 6]])
>>> x * z
Tensor(data=
[[ 1  4  9]
 [ 4 10 18]], requires_grad=False)
```
### Automatic backpropagation
But a `Tensor` is not just an `ndarray`, they also keep track of their own gradient.
```python
>>> speed = Tensor(1, requires_grad=True)
>>> time = Tensor(10, requires_grad=True)
>>> distance = speed * time
>>> distance
Tensor(data=10, requires_grad=True)
```
If a `Tensor` is created as the result of a `Tensor` operation involving a `Tensor` with `requires_grad=True`,
the resulting `Tensor` will be able to backpropagate it's own gradient to it's ancestor.
```python
>>> distance.backward()
>>> speed.grad
array(10.)
>>> time.grad
array(1.)
```
By calling the `backward` method on `distance` the gradient of `speed` and `time` is automatically updated.
We can see that increasing `speed` by 1 would result in an increase in `distance` by 10, while an increase
in `time` by 1 would only increase `distance` by 1.

The `Tensor` class supports backpropagation over arbitrary compositions of `Tensor` operations.
```python
>>> t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
>>> t2 = Tensor([[1], [2], [3]])
>>> t3 = t1 @ t2 + 1
>>> t4 = t3 * 7
>>> t5 = t4.sum()
>>> t5.backward()
>>> t1.grad
array([[ 7., 14., 21.],
       [ 7., 14., 21.]])
```
### The `Module` API
`gustavgrad` provides some tools to simplify setting up and training machine learning models.
The `Module` API makes it easier to manage multiple related `Tensors` by registering them as
`Parameters`.
A `Parameter` is just a randomly initialized `Tensor`.
```python
from gustavgrad.module import Module, Parameter
from gustavgrad.function import tanh
class MultilayerPerceptron(Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 100) -> None:
        self.layer1 = Parameter(input_size, hidden_size)
        self.bias1 = Parameter(hidden_size)
        self.layer2 = Parameter(hidden_size, output_size)
        self.bias2 = Parameter(output_size)
        
    def predict(self, x: Tensor) -> Tensor:
        x = x @ self.layer1 + self.bias1
        x = tanh(x)
        x = x @ self.layer2 + self.bias2
        return x
```
By subclassing `Module` our `MultilayerPerceptron` class automatically gets some helper methods
for managing its `Parameters`.
Let's create a `MultilayerPerceptron` that tries to learn the XOR function.
```python
xor_input = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_targets = Tensor([[0], [1], [1], [0]])
xor_mlp = MultilayerPerceptron(input_size=2, output_size=1, hidden_size=4)
```
We can use the model to make predictions on the `xor_input` `Tensor`.
```python
>>> predictions = xor_mlp.predict(xor_input)
>>> predictions
Tensor(data=
[[-1.79888385]
 [-1.07965756]
 [ 0.34373135]
 [ 1.63366069]], requires_grad=True)
```
The predictions of the randomly initialized model aren't right, but we can improve the model by
calculating the gradient of it's `Parameters` in respect to a loss function.
```python
from gustavgrad.loss import SquaredErrorLoss
se_loss = SquaredErrorLoss()
loss = se_loss.loss(xor_targets, predictions)
loss.backward()
```
`loss` is a `Tensor`, so we can call its `backward` method to do backpropagation through our `xor_mlp`.
We can then adjust the weights of all `Parameters` in `xor_mlp` using gradient descent:
```python
from gustavgrad.optim import SGD
optim = SGD(lr=0.01)
optim.step(xor_mlp)
```
After updating the weights we can reset the gradients of all parameters and make new predictions:
```python
>>> xor_mlp.zero_grad()
>>> predictions = xor_mlp.predict(xor_input)
>>> predictions
Tensor(data=
[[-1.51682686]
 [-0.78583272]
 [ 0.55994602]
 [ 1.67962174]], requires_grad=True)
```
See [examples/xor.py](examples/xor.py) for a full example of how `gustavgrad` can be used to learn the XOR function.
The [examples](examples/) directory also contains some other basic examples of how the library can be used.
