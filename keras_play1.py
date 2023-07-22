from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow as tf

layer = layers.Dense(
    units= 64,
    kernel_regularizer = regularizers.L1L2(l1=1e-5,l2=1e-4),
    bias_regularizer = regularizers.L2(1e-4),
    activity_regularizer = regularizers.L2(1e-5)
)

# print(layer)
tensor = tf.ones(shape=(5, 5)) * 2.0
out = layer(tensor)
# print(len(out))
# print(out)


# printing the losses
# print(tf.math.reduce_sum(layer.losses))


dense = tf.keras.layers.Dense(3, kernel_regularizer="l1")
# print(dense)

dense = tf.keras.layers.Dense(3, kernel_regularizer="l2")

dense = tf.keras.layers.Dense(3, kernel_regularizer="l1_l2")


regularizer = tf.keras.regularizers.OrthogonalRegularizer(factor= 0.01)
# print(regularizer)

layer = tf.keras.layers.Dense(units=4, kernel_regularizer = regularizer)
# print(layer)


# creating custom regularizers
def my_regularizer(x):
    return 1e-3 * tf.reduce_sum(tf.square(x))

# print(my_regularizer(tensor))

my_regularizer(tensor)


class Myregularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, strength):
        self.strength = strength

    def __call__(self, x):
        return self.strength * tf.reduce_sum(tf.square(x))



class MyRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, strength):
        self.strength = strength

    def __call__(self, x):
        return self.strength * tf.reduce_sum(tf.square(x))

    def get_config(self):
        return {'strength': self.strength}


# using the class here
my_reg_instance = Myregularizer(strength = 0.001)

layer = tf.keras.layers.Dense(units=4, kernel_regularizer= my_reg_instance)

out = layer(tensor)

print(out)

print(tf.math.reduce_sum(layer.losses))

my_reg_instance = MyRegularizer(strength= 0.01)

layer = tf.keras.layers.Dense(units= 4, kernel_regularizer=my_reg_instance)

out = layer(tensor)

print(out)

print(tf.math.reduce_sum(layer.losses))