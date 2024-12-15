import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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




class OptimizedKNN:
    """
    Optimized K-Nearest Neighbors classifier with parallel processing capabilities.
    
    Features:
    - Multiple distance metrics (Euclidean, Manhattan, Minkowski)
    - Parallel processing for faster predictions
    - Efficient distance calculations using vectorization
    - Distance caching for repeated predictions
    - Support for weighted voting
    
    Parameters:
    -----------
    n_neighbors : int, default=5
        Number of neighbors to use for prediction
    metric : str, default="euclidean"
        Distance metric to use. Options: "euclidean", "manhattan", "minkowski"
    p : int, optional
        Power parameter for Minkowski metric
    weights : str, default="uniform"
        Weight function used in prediction. Options: "uniform", "distance"
    n_jobs : int, default=1
        Number of parallel jobs. If -1, use all available CPU cores
    """
    
    def __init__(self, n_neighbors=5, metric="euclidean", p=None, 
                 weights="uniform", n_jobs=1):
        
        if not isinstance(n_neighbors, int) or n_neighbors <= 0:
            raise ValueError(f"n_neighbors must be a positive integer, got {n_neighbors}")
            
        if metric not in ["euclidean", "manhattan", "minkowski"]:
            raise ValueError(f"Unsupported metric: {metric}")
            
        if weights not in ["uniform", "distance"]:
            raise ValueError(f"Unsupported weight function: {weights}")
            
        if metric == "manhattan" and p not in (None, 1):
            raise ValueError("When metric='manhattan', p should be None or 1")
            
        if metric == "euclidean" and p not in (None, 2):
            raise ValueError("When metric='euclidean', p should be None or 2")
            
        if metric == "minkowski" and p is None:
            raise ValueError("For metric='minkowski', p parameter is required")
        
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.weights = weights
        self.n_jobs = n_jobs if n_jobs != -1 else None  
        
        
        self.X = None
        self.y = None
        self._distance_cache = {}
        
    def _validate_data(self, X, y=None):
        """Validate and convert input data to numpy arrays."""
        X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)
            if len(X) != len(y):
                raise ValueError("X and y must have the same number of samples")
        return X, y
    
    def fit(self, X, y):
        """
        Fit the KNN model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : OptimizedKNN
            The fitted model
        """
        self.X, self.y = self._validate_data(X, y)
        self._classes = np.unique(y)
        self._distance_cache.clear()  
        return self
    
    def _compute_distances(self, x):
        """Compute distances between a single point and all training points."""
        if self.metric == "manhattan":
            return np.sum(np.abs(self.X - x), axis=1)
        elif self.metric == "euclidean":
            
            return np.sqrt(np.sum((self.X - x) ** 2, axis=1))
        else:  
            return np.power(np.sum(np.power(np.abs(self.X - x), self.p), axis=1), 1/self.p)
    
    def _predict_single(self, x):
        """Predict class for a single sample."""
        # Check cache first
        x_key = hash(x.tobytes())
        if x_key in self._distance_cache:
            distances = self._distance_cache[x_key]
        else:
            distances = self._compute_distances(x)
            self._distance_cache[x_key] = distances
        
        # Get nearest neighbors
        nearest_indices = np.argpartition(distances, self.n_neighbors)[:self.n_neighbors]
        nearest_distances = distances[nearest_indices]
        nearest_labels = self.y[nearest_indices]
        
        if self.weights == "uniform":
            # Simple majority voting
            return Counter(nearest_labels).most_common(1)[0][0]
        else:  # distance weighting
            # Avoid division by zero
            weights = 1 / (nearest_distances + np.finfo(float).eps)
            weighted_votes = {}
            for label, weight in zip(nearest_labels, weights):
                weighted_votes[label] = weighted_votes.get(label, 0) + weight
            return max(weighted_votes.items(), key=lambda x: x[1])[0]
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
        
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        X, _ = self._validate_data(X)
        
        if self.n_jobs == 1:
            return np.array([self._predict_single(x) for x in X])
        
        # Parallel prediction
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            predictions = list(executor.map(self._predict_single, X))
        return np.array(predictions)
    
    def get_params(self):
        """Get model parameters."""
        return {
            'n_neighbors': self.n_neighbors,
            'metric': self.metric,
            'p': self.p,
            'weights': self.weights,
            'n_jobs': self.n_jobs
        }