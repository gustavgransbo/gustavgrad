# Autograd
An autograd library built on NumPy. 

Inspired by [Joel Grus's livecoding](https://github.com/joelgrus/autograd/tree/master), I also want to try to code an autograd library. 

The idea behind autograd is to define a Tensor class, and a set of arithmetic operations on tensors, which we know how to calculate the first order derivative for.
Using the chain-rule, the gradient of the composition of multiple operations can be calculated, since we know how to calculate the first order derivative of the basic operations.

By reproducing Joel's work, I hope to learn more about:
* Automatic Differentiation
* Pytest
* Developing clean Python APIs
