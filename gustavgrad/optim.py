" Optimizers for gustavgrad Modules "

from gustavgrad.module import Module


class SGD:
    " Stochastic Gradient Descent Optimizer"

    def __init__(self, lr: float = 0.001) -> None:
        self.lr = lr

    def step(self, module: Module) -> None:
        for parameter in module.parameters():
            if parameter.grad is not None:
                parameter -= parameter.grad * self.lr
