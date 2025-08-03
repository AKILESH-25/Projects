from sklearn.decomposition import PCA
class PCAFeatureExtractor:
    def __init__(self, n_components=100):
        self.n_components = n_components
        self.model = PCA(n_components=n_components)

    def fit_transform(self, X):
        return self.model.fit_transform(X)

    def transform(self, X):
        return self.model.transform(X)

    def explained_variance(self):
        return self.model.explained_variance_ratio_
