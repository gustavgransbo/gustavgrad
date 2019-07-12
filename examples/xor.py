""" Example of how to learn the XOR function using the autograd library """

from autograd import Tensor
from autograd.functions import tanh
import numpy as np

X = Tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# one-hot encoded labels
y = Tensor([
    [1, 0],
    [0, 1],
    [1, 0],
    [1, 0]
])

class Model:
    """ A multi layer perceptron that should learn the XOR function """
    def __init__(self) -> None:
        self.layer1 = Tensor(np.random.randn(2, 4), requires_grad=True)
        self.bias1 = Tensor(np.random.randn(4), requires_grad=True)
        self.layer2 = Tensor(np.random.randn(4, 2), requires_grad=True)
        self.bias2 = Tensor(np.random.randn(2), requires_grad=True)

    def predict(self, x: Tensor) -> Tensor:
        x = x@self.layer1 + self.bias1
        x = tanh(x)
        x = x@self.layer2 + self.bias2
        return x

    def zero_grad(self) -> None:
        self.layer1.zero_grad()
        self.layer2.zero_grad()
        self.bias1.zero_grad()
        self.bias2.zero_grad()

    def sgd_step(self, lr: float = 0.001) -> None:
        self.layer1 -= self.layer1.grad * lr
        self.layer2 -= self.layer2.grad * lr
        self.bias1 -= self.bias1.grad * lr
        self.bias2 -= self.bias2.grad * lr

epochs = 1000
lr = 0.01
mlp = Model()

for _ in range(epochs):

    mlp.zero_grad()

    pred = mlp.predict(X)
    error = y - pred
    mse_loss = (error * error).sum()
    mse_loss.backward()

    mlp.sgd_step(lr)
    
    print(mse_loss.data)

print(pred.data
)

        
