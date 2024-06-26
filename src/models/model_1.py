import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class MyModel(Model):
    def __init__(self, input_shape):
        super(MyModel, self).__init__()
        self.dense1 = Dense(10, activation='relu', input_shape=(input_shape,))
        self.dense2 = Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.dense1(x)
        return self.dense2(x)
