"""
Example of how a multivariate linear regression problem can be solved with the package.
"""

import numpy as np
from autograd import Tensor

# 100 training examples with 3 features
x = Tensor(np.random.rand(100, 3))

# The function we want to learn
coefs = Tensor(np.asarray([1., 3., 5.]))
bias = 2
y = x @ coefs + bias

# Our model
w = Tensor(np.random.randn(3), requires_grad=True)
b = Tensor(np.random.rand(), requires_grad = True)

# Train the model
lr = 0.001
for _ in range(1000):

    w.zero_grad(), b.zero_grad()

    pred = x @ w + b
    errors = (y - pred)
    mse_loss = (errors * errors).sum()
    mse_loss.backward()

    print(mse_loss.data)

    # Gradient Descent
    w.data -= lr * w.grad
    b.data -= lr * b.grad

print(f"Target function: coefficients={coefs.data}, bias={bias}")
print(f"w={w.data}, b={b.data}")