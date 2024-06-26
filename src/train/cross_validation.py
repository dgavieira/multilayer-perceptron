import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime

class CrossValidator:
    def __init__(self, model_creator, n_splits=5, random_state=42):
        self.model_creator = model_creator
        self.kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.scores = []

    @classmethod
    def get_log_dir(cls, model_name):
        log_dir = "logs/fit/" + model_name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return log_dir

    def cross_validate(self, X, y, epochs=50, batch_size=10):
        for train_index, test_index in self.kfold.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = self.model_creator()
            log_dir = self.get_log_dir('model')
            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[tensorboard_callback])
            score = model.evaluate(X_test, y_test, verbose=1)
            self.scores.append(score)

    def mean_accuracy(self):
        return np.mean(self.scores, axis=0)[1]

# Assuming create_model_1 is already defined
def create_model_1():
    model = Sequential([
        Dense(10, activation='relu', input_shape=(X.shape[1],)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Example usage
# Assuming X and y are already defined and are numpy arrays
cross_validator = CrossValidator(create_model_1)
cross_validator.cross_validate(X, y)
print('Arquitetura 1 - Acurácia Média:', cross_validator.mean_accuracy())
