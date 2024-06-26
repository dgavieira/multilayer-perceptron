import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class MyModel2(Model):
    def __init__(self, input_shape):
        super(MyModel2, self).__init__()
        self.dense1 = Dense(20, activation='relu', input_shape=(input_shape,))
        self.dense2 = Dense(10, activation='relu')
        self.dense3 = Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)