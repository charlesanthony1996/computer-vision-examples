import time
import sympy

from sympy.tensor.tensor import TensorIndexType, TensorHead
from sympy.tensor.toperators import PartialDerivative
from sympy import symbols
import numpy as np
import torch

L = TensorIndexType("L")
A = TensorHead("A", [L])
B = TensorHead("B", [L])

i, j, k = symbols("i j k")

print(symbols("i j k"))
print(A)
print(L)

