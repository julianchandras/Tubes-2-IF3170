import numpy as np

class KNeighborsClassifier():
    def __init__(self, n_neighbors=5, metric="euclidean", p=None):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p

        if not isinstance(n_neighbors, int):
            raise TypeError(f"n_neighbors must be an integer, got {type(n_neighbors).__name__} instead.")

        if self.metric == "manhattan" and self.p not in (None, 1):
            raise ValueError("when metric='manhattan', p should be None or 1.")
        if self.metric == "euclidean" and self.p not in (None, 2):
            raise ValueError("when metric='euclidean', p should be None or 2.")
        if metric == "minkowski" and p is None:
            raise ValueError("For metric='minkowski', you must specify a value for p.")

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        if self.metric == "manhattan":
            dist = np.sum(np.abs(self.X - x), axis=1)
        elif self.metric == "euclidean":
            dist = np.linalg.norm(self.X - x, axis=1)
        elif self.metric == "minkowski":
            dist = np.sum(np.abs(self.X - x) ** self.p, axis=1) ** (1 / self.p)

        nearest_indices = np.argsort(dist)[:self.n_neighbors]
        nearest_labels = self.y[nearest_indices]
        prediction = np.bincount(nearest_labels).argmax()
        
        return prediction
