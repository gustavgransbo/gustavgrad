import inspect
from typing import Iterator

import numpy as np

from gustavgrad.tensor import Tensor


class Parameter(Tensor):
    " A Parameter is used to register a Tensor as part of a Module "

    def __init__(self, *shape: int) -> None:
        # TODO: Add possibility to use custom initialization schemes
        super().__init__(np.random.randn(*shape), requires_grad=True)


class Module:
    " A Module wraps multiple Parameters in a single computational block "

    def parameters(self) -> Iterator[Parameter]:
        for _, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()
