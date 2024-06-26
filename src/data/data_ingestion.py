from ucimlrepo import fetch_ucirepo

class MammographicMassDataset:
    def __init__(self, dataset_id=161):
        self.dataset_id = dataset_id
        self.dataset = None
        self.features = None
        self.targets = None
        self.metadata = None
        self.variables = None
        self._fetch_data()

    def _fetch_data(self):
        self.dataset = fetch_ucirepo(id=self.dataset_id)
        self.features = self.dataset.data.features
        self.targets = self.dataset.data.targets
        self.metadata = self.dataset.metadata
        self.variables = self.dataset.variables

    def get_features(self):
        return self.features

    def get_targets(self):
        return self.targets

    def get_metadata(self):
        return self.metadata

    def get_variables(self):
        return self.variables

# Example usage
mammographic_mass_data = MammographicMassDataset()

# Fetch features and targets
X = mammographic_mass_data.get_features()
y = mammographic_mass_data.get_targets()

# Print metadata and variable information
print(mammographic_mass_data.get_metadata())
print(mammographic_mass_data.get_variables())
