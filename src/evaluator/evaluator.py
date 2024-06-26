import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

class ModelEvaluator:
    def __init__(self, model_creator, n_splits=5, random_state=42):
        self.model_creator = model_creator
        self.kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.metrics = {'accuracy': [], 'sensitivity': [], 'specificity': [], 'precision': [], 'f1_score': []}

    def evaluate(self, X, y, epochs=50, batch_size=10):
        for train_index, test_index in self.kfold.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = self.model_creator()
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            y_pred = (model.predict(X_test) > 0.5).astype("int32")

            report = classification_report(y_test, y_pred, output_dict=True)

            self.metrics['accuracy'].append(report['accuracy'])
            self.metrics['sensitivity'].append(report['1']['recall'])
            self.metrics['specificity'].append(report['0']['recall'])
            self.metrics['precision'].append(report['1']['precision'])
            self.metrics['f1_score'].append(report['1']['f1-score'])

    def print_mean_metrics(self):
        for metric, values in self.metrics.items():
            print(f'{metric.capitalize()} Média:', np.mean(values))

# Example usage
def create_model_1():
    model = Sequential([
        Dense(10, activation='relu', input_shape=(X.shape[1],)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_model_2():
    model = Sequential([
        Dense(20, activation='relu', input_shape=(X.shape[1],)),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Assuming X and y are already defined and are numpy arrays
evaluator_1 = ModelEvaluator(create_model_1)
evaluator_1.evaluate(X, y)
evaluator_1.print_mean_metrics()

evaluator_2 = ModelEvaluator(create_model_2)
evaluator_2.evaluate(X, y)
evaluator_2.print_mean_metrics()
