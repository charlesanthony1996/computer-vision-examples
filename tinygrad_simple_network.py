import numpy as np
import time

from tinygrad.tensor import Tensor

t1 = Tensor([1, 2, 3, 4])
na = np.array([1, 2, 3, 4])

t2 = Tensor(na)

# print(t2)

full = Tensor.full(shape=(2, 3), fill_value=5)

print(full)

zeros = Tensor.zeros(2, 3)

print(zeros)

ones = Tensor.ones(2, 3)

print(ones)

full_like = Tensor.full_like(full, fill_value=2)
print(full_like)

zeros_like = Tensor.zeros_like(full)

print(zeros_like)

ones_like = Tensor.ones_like(full)

print(ones_like)

eye = Tensor.eye(3)
print(eye)

arange = Tensor.arange(start= 0, stop=10, step= 1)
print(arange)


from tinygrad.helpers import dtypes

t3 = Tensor([1, 2, 3, 4, 5], dtype=dtypes.int32)
print(t3)

t4 = Tensor([1, 2, 3, 4, 5])
t5 = (t4 + 1) * 2
t6 = (t5 + t4).relu().log_softmax()

print(t6.numpy())

print(t5.numpy())


# building a small model

# basically how the linear class is built

# class Linear:
#     def __init__(self, in_features, out_features, bias=True, initialization: str='kaiming_uniform'):
#         self.weight = getattr(Tensor, initialization)(out_features, in_features)
#         self.bias = Tensor.zeros(out_features) if bias else None

#     def __call__(self, x):
#         return x.linear(self.weight.transpose(), self.bias)



from tinygrad.nn import Linear

# print(Linear)

class TinyNet: 
    def __init__(self):
        self.l1 = Linear(784, 128, bias=False)
        self.l2 = Linear(128, 10, bias=False)


    def __call__(self, x):
        x = self.l1(x)
        x = x.leakyrelu()
        x = self.l2(x)
        return x.log_softmax()

net = TinyNet()

print(net)

# this flag has to be made true
Tensor.training = True

# from extra.training import sparse_categorical_crossentropy

def cross_entropy(out, Y):
    num_classes = out.shape[-1]
    YY = Y.flatten().astype(np.int32)
    y = np.zeros((YY.shape[0], num_classes), np.float32)
    y[range(y.shape[0]), YY] = -1.0 * num_classes
    y = y.reshape(list(Y.shape) + [num_classes])
    y = Tensor(y)
    return out.mul(y).mean()

    










