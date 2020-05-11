""" 
Since this autograd project was inspired by Joel Grus, it feels only right that 
I use it to solve his go-to problem: FizzBuzz

The FizzBuzz problem tasks a prospective software engineer to develop a program that 
for takes all the numbers from 1 to 100, and prints them with a simple twist.
If a number is divisible by 3 you print Fizz, if it's divisible by 5 you print Buzz,
and if it's divisible by 15 you print FizzBuzz. All other numbers are printed as are.

Joel's aproach to solve this with a neural net builds upon taking the 10 bit binary-
encodings of numbers as input, and outputing a one-hot encoded label corresponding to 
the four classes. Training is done on the numbers 100-1023.
"""

from typing import List
import time

from autograd import Tensor
from autograd.functions import tanh
from autograd.loss import LogitBinaryCrossEntropy
from autograd.module import Module, Parameter

import numpy as np
from tqdm import tqdm


def binary_encode(x: int) -> List[int]:
    """ Binary encode x using 10 binary digits"""
    return [x >> i & 1 for i in range(10)]

def fizz_buzz_encode(x: int) -> List[int]:
    """ One-hot encode an integer according to it's FizzBuzz class """
    if x % 15 == 0:
        return [0,0,0,1]
    elif x % 5 == 0:
        return [0,0,1,0]
    elif x % 3 == 0:
        return [0,1,0,0]
    else:
        return [1,0,0,0]

X_train = Tensor(np.asarray([binary_encode(x) for x in range(101, 1024)]))
y_train = Tensor(np.asarray([fizz_buzz_encode(x) for x in range(101, 1024)]))

class Model(Module):
    """ A multi layer perceptron that should learn FizzBuzz function """
    def __init__(self, num_hidden: int = 100) -> None:
        self.layer1 = Parameter(10, 100)
        self.bias1 = Parameter(num_hidden)
        self.layer2 = Parameter(num_hidden, 4)
        self.bias2 = Parameter(4)

    def predict(self, x: Tensor) -> Tensor:
        x = x@self.layer1 + self.bias1
        x = tanh(x)
        x = x@self.layer2 + self.bias2
        return x

    def sgd_step(self, lr: float = 0.001) -> None:

        for parameter in self.parameters():
            parameter -= parameter.grad * lr

epochs = 10_000
lr = 0.01
mlp = Model()
# Ideally I would use cross-entropy and a softmax layer since the targets are mutually exclusive,
# but I haven't implemented cross entropy loss.
bce_loss = LogitBinaryCrossEntropy()
idx = np.arange(X_train.shape[0])
batch_size = 64
t1 = time.time()
for _ in tqdm(range(epochs)):

    for start in range(0, X_train.shape[0], batch_size):

        X = X_train[idx[start:start+batch_size]]
        y = y_train[idx[start:start+batch_size]]

        mlp.zero_grad()

        pred = mlp.predict(X)
        loss = bce_loss.loss(y, pred)
        loss.backward()

        mlp.sgd_step(lr)
    np.random.shuffle(idx)

# Includes 0, even though we wont actually evaluate on 0
X_test = Tensor([binary_encode(x) for x in range(0, 101)])
y_test = Tensor([fizz_buzz_encode(x) for x in range(0, 101)]) 

correct = 0
for i in range(1, 101):
    labels = [i, 'Fizz', 'Buzz', 'FizzBuzz']
    pred = mlp.predict(X_test[i])
    pred_idx = pred.data.argmax()

    print(f"{i}: {labels[pred_idx]}, {labels[y_test[i].data.argmax()]}")
    correct += y_test[i].data.argmax() == pred_idx

print(f"Correct: {correct} / 100")
print(f"Time taken (train+evaluate): {time.time()- t1}s")

"""
Output after training for 10,000 epochs:
1: 1, 1
2: 2, 2
3: Fizz, Fizz
4: 4, 4
5: Buzz, Buzz
6: Fizz, Fizz
7: 7, 7
8: 8, 8
9: Fizz, Fizz
10: Buzz, Buzz
11: 11, 11
12: Fizz, Fizz
13: 13, 13
14: 14, 14
15: FizzBuzz, FizzBuzz
16: 16, 16
17: 17, 17
18: Fizz, Fizz
19: 19, 19
20: Buzz, Buzz
21: Fizz, Fizz
22: 22, 22
23: 23, 23
24: Fizz, Fizz
25: Buzz, Buzz
26: 26, 26
27: Fizz, Fizz
28: 28, 28
29: 29, 29
30: FizzBuzz, FizzBuzz
31: 31, 31
32: 32, 32
33: Fizz, Fizz
34: 34, 34
35: Buzz, Buzz
36: Fizz, Fizz
37: 37, 37
38: 38, 38
39: Fizz, Fizz
40: Buzz, Buzz
41: 41, 41
42: Fizz, Fizz
43: 43, 43
44: 44, 44
45: FizzBuzz, FizzBuzz
46: 46, 46
47: 47, 47
48: Fizz, Fizz
49: 49, 49
50: Buzz, Buzz
51: Fizz, Fizz
52: 52, 52
53: 53, 53
54: Fizz, Fizz
55: Buzz, Buzz
56: 56, 56
57: Fizz, Fizz
58: 58, 58
59: 59, 59
60: FizzBuzz, FizzBuzz
61: 61, 61
62: 62, 62
63: Fizz, Fizz
64: 64, 64
65: Buzz, Buzz
66: Fizz, Fizz
67: 67, 67
68: 68, 68
69: Fizz, Fizz
70: Buzz, Buzz
71: 71, 71
72: Fizz, Fizz
73: 73, 73
74: 74, 74
75: FizzBuzz, FizzBuzz
76: 76, 76
77: 77, 77
78: Fizz, Fizz
79: 79, 79
80: Buzz, Buzz
81: Fizz, Fizz
82: 82, 82
83: 83, 83
84: Fizz, Fizz
85: Buzz, Buzz
86: 86, 86
87: Fizz, Fizz
88: 88, 88
89: 89, 89
90: FizzBuzz, FizzBuzz
91: 91, 91
92: 92, 92
93: Fizz, Fizz
94: 94, 94
95: Buzz, Buzz
96: Fizz, Fizz
97: 97, 97
98: 98, 98
99: Fizz, Fizz
100: Buzz, Buzz
Correct: 100 / 100
Time taken (train+evaluate): 313.07683658599854s
"""
