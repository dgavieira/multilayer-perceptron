from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class DataProcessor:
    def __init__(self, dataset_id):
        self.dataset = fetch_ucirepo(id=dataset_id)
        self.X = self.dataset.data.features
        self.y = self.dataset.data.targets

    def select_features(self, features, target):
        self.X = self.X[features]
        self.y = self.y[target]

    def handle_missing_values(self):
        imputer = SimpleImputer(strategy='mean')
        self.X = imputer.fit_transform(self.X)

    def standardize_data(self):
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)