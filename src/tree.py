import numpy as np
from scipy.stats import mode

class Node():
    """
    Ref:
    https://www.youtube.com/watch?v=mN7i0U4YMqY
    https://www.youtube.com/watch?v=TDkZev5xjfg
    """
    def __init__(self, X, y, depth, max_features):
        self.depth = depth
        self.X = X
        self.y = y
        self.max_features = max_features
        self.best_feature_index = None
        self.best_threshold = None
        self.left = None
        self.right = None
        classes, class_counts = np.unique(self.y, return_counts=True)
        self.label = classes[np.argmax(class_counts)]

    def make_split(self, max_depth, min_samples_split):
        self.best_feature_index, self.best_threshold = self._find_best_split()
        if self.best_feature_index is not None:
            if self.depth < max_depth and self.X.shape[0] > min_samples_split and not(np.all(self.y == self.y[0])):
                left_mask = self.X[:, self.best_feature_index] < self.best_threshold
                right_mask = ~left_mask

                if np.sum(left_mask) > 0 and np.sum(right_mask) > 0:
                    self.left = Node(self.X[left_mask], self.y[left_mask], depth=self.depth+1, max_features=self.max_features)
                    self.right = Node(self.X[right_mask], self.y[right_mask], depth=self.depth+1, max_features=self.max_features)

                    self.left.make_split(max_depth, min_samples_split)
                    self.right.make_split(max_depth, min_samples_split)

    def _calculate_information_gain(self, feature_index, threshold):
        left_mask = self.X[:, feature_index] < threshold
        right_mask = ~left_mask

        left_entropy = self._calculate_entropy(self.y[left_mask])
        right_entropy = self._calculate_entropy(self.y[right_mask])

        n_left = np.sum(left_mask)
        n_right = self.y.shape[0] - n_left

        weighted_entropy = (n_left / len(self.y)) * left_entropy + (n_right / len(self.y)) * right_entropy
        return self._calculate_entropy(self.y) - weighted_entropy

    def _calculate_entropy(self, y):
        _, class_counts = np.unique(y, return_counts=True)
        probabilities = class_counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _find_best_split(self):
        largest_info_gain = 0
        best_feature_index = None
        best_threshold = None

        selected_features = np.random.choice(self.X.shape[1], self.max_features, replace=False)
        for feature_index in selected_features:
            sorted_values = np.sort(np.unique(self.X[:, feature_index]))
            thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2

            for threshold in thresholds:
                info_gain = self._calculate_information_gain(feature_index, threshold)
                if info_gain > largest_info_gain:
                    best_feature_index = feature_index
                    best_threshold = threshold
                    largest_info_gain = info_gain

        return best_feature_index, best_threshold

class DecisionTreeClassifier():
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.classes_ = None
        self.n_classes_ = None
        self.root = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        if self.max_features is None:
            self.max_features = X.shape[1]
        self.root = Node(X, y, depth=0, max_features=self.max_features)
        self.root.make_split(self.max_depth, self.min_samples_split)
        
    def _predict_single(self, x):
        node = self.root
        while node.left and node.right:
            if node.best_feature_index is None:
                break

            if x[node.best_feature_index] < node.best_threshold:
                node = node.left
            else:
                node = node.right
        return node.label

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])
    
class RandomForestClassifier():
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2):
        self.n_estimaors = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = None
        self.trees = []

    def fit(self, X, y):
        self.max_features = np.floor(np.sqrt(X.shape[1]))
        n_samples = X.shape[0]

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([mode(preds).mode[0] for preds in tree_preds.T])
