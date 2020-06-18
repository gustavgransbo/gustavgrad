" Optimizers for autograd Modules "

from autograd.module import Module


class SGD:
    " Stochastic Gradient Descent Optimizer"

    def __init__(self, lr=0.001) -> None:
        self.lr = lr

    def step(self, module: Module) -> None:
        for parameter in module.parameters():
            # FIXME: Figure out typing of inherited __isub__
            parameter -= parameter.grad * self.lr
