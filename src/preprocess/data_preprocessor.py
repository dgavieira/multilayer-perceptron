from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

class DataPreprocessor:
    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()

    def preprocess_features(self, X):
        X_selected = X[['Age', 'Shape', 'Margin', 'Density']]
        X_imputed = self.imputer.fit_transform(X_selected)
        X_scaled = self.scaler.fit_transform(X_imputed)
        return X_scaled

    def preprocess_target(self, y):
        return y['Severity']

# Example usage
# Assuming X and y are already defined and are pandas DataFrames
data_preprocessor = DataPreprocessor(features=['Age', 'Shape', 'Margin', 'Density'], target='Severity')

# Preprocess features and target
X_preprocessed = data_preprocessor.preprocess_features(X)
y_preprocessed = data_preprocessor.preprocess_target(y)

# X_preprocessed and y_preprocessed are now ready for use
print(X_preprocessed)
print(y_preprocessed)
