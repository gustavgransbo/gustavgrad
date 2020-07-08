# gustavgrad
[![Tests](https://github.com/gustavgransbo/gustavgrad/workflows/Tests/badge.svg)](https://github.com/gustavgransbo/gustavgrad/actions?workflow=Tests)
[![codecov](https://codecov.io/gh/gustavgransbo/gustavgrad/branch/master/graph/badge.svg)](https://codecov.io/gh/gustavgransbo/gustavgrad)

An autograd library built on NumPy, inspired by [Joel Grus's livecoding](https://github.com/joelgrus/autograd/tree/master).

The idea behind gustavgrad is to define a Tensor class, and a set of arithmetic operations on tensors, which we know how to calculate the first order derivative for.
Using the chain-rule, the gradient of the composition of multiple operations can be calculated, since we know how to calculate the first order derivative of the basic operations.
