"""
Example of how to learn the XOR function using the gustavgrad library
"""

from gustavgrad import Tensor
from gustavgrad.function import tanh
from gustavgrad.loss import SquaredErrorLoss
from gustavgrad.module import Module, Parameter
from gustavgrad.optim import SGD

xor_input = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_targets = Tensor([[0], [1], [1], [0]])


class MultilayerPerceptron(Module):
    """ A multilayer perceptron with two layers """

    def __init__(
        self, input_size: int, output_size: int, hidden_size: int = 100
    ) -> None:
        self.layer1 = Parameter(input_size, hidden_size)
        self.bias1 = Parameter(hidden_size)
        self.layer2 = Parameter(hidden_size, output_size)
        self.bias2 = Parameter(output_size)

    def predict(self, x: Tensor) -> Tensor:
        x = x @ self.layer1 + self.bias1
        x = tanh(x)
        x = x @ self.layer2 + self.bias2
        return x


epochs = 1000
optim = SGD(lr=0.01)
xor_mlp = MultilayerPerceptron(input_size=2, output_size=1, hidden_size=4)
se_loss = SquaredErrorLoss()

for _ in range(epochs):

    xor_mlp.zero_grad()

    prediction = xor_mlp.predict(xor_input)
    loss = se_loss.loss(xor_targets, prediction)
    loss.backward()

    optim.step(xor_mlp)

    print(loss.data)

print(prediction.data.round(4))
