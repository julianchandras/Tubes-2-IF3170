import numpy as np
from collections import Counter
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union, Tuple
import threading

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  
        self.threshold = threshold  
        self.left = left 
        self.right = right  
        self.value = value 

class ID3DecisionTree:
    def __init__(self, max_depth: Optional[int] = None, 
                 min_samples_split: int = 2,
                 n_jobs: int = -1):
        """
        Initialize OptimizedID3DecisionTree classifier
        
        Parameters:
        -----------
        max_depth : int, optional
            Maximum depth of the tree
        min_samples_split : int
            Minimum samples required to split a node
        n_jobs : int
            Number of parallel jobs. -1 means using all processors
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs if n_jobs > 0 else threading.active_count()
        self.root = None
        self.n_classes = None
        self.feature_types = None
        self.feature_names = None
        self._lock = threading.Lock()

    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_types: Optional[List[str]] = None,
            feature_names: Optional[List[str]] = None) -> 'ID3DecisionTree':
        """
        Train the decision tree using parallel processing
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        feature_types : list of str, optional
            List of feature types ('numerical' or 'categorical')
        feature_names : list of str, optional
            List of feature names
        """
        X = np.array(X)
        y = np.array(y)
        
        self.n_classes = len(np.unique(y))
        self.feature_types = feature_types or ['numerical'] * X.shape[1]
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        self.root = self._grow_tree(X, y)
        return self

    def _entropy(self, y: np.ndarray) -> float:
        """Calculate entropy of a node using vectorized operations"""
        hist = np.bincount(y)
        ps = hist / len(y)
        ps = ps[ps > 0]
        return -np.sum(ps * np.log2(ps))

    def _information_gain(self, y: np.ndarray, y_left: np.ndarray, 
                         y_right: np.ndarray) -> float:
        """Calculate information gain for a split using vectorized operations"""
        if len(y_left) == 0 or len(y_right) == 0:
            return 0
            
        n = len(y)
        parent_entropy = self._entropy(y)
        n_l, n_r = len(y_left), len(y_right)
        child_entropy = (n_l / n) * self._entropy(y_left) + (n_r / n) * self._entropy(y_right)
        return parent_entropy - child_entropy

    def _find_best_split_for_feature(self, X: np.ndarray, y: np.ndarray, 
                                   feature_idx: int) -> Tuple[float, Optional[Union[float, str]]]:
        """Find the best split for a single feature"""
        best_gain = -1
        best_threshold = None
        
        if self.feature_types[feature_idx] == 'numerical':
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                if not (np.any(left_mask) and np.any(~left_mask)):
                    continue
                    
                gain = self._information_gain(y, y[left_mask], y[~left_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold
        else:
            unique_values = np.unique(X[:, feature_idx])
            for value in unique_values:
                left_mask = X[:, feature_idx] == value
                if not (np.any(left_mask) and np.any(~left_mask)):
                    continue
                    
                gain = self._information_gain(y, y[left_mask], y[~left_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = value
                    
        return best_gain, best_threshold

    def _parallel_find_best_split(self, X: np.ndarray, y: np.ndarray, 
                                feature_indices: List[int]) -> List[Tuple[int, float, Optional[Union[float, str]]]]:
        """Find best splits for multiple features in parallel"""
        results = []
        for idx in feature_indices:
            gain, threshold = self._find_best_split_for_feature(X, y, idx)
            results.append((idx, gain, threshold))
        return results

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Recursively grow the decision tree using parallel processing for feature selection"""
        n_samples, n_features = X.shape
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            return Node(value=Counter(y).most_common(1)[0][0])
        
        # Split features into chunks for parallel processing
        feature_chunks = np.array_split(range(n_features), self.n_jobs)
        
        # Find best split using parallel processing
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            future_results = [
                executor.submit(self._parallel_find_best_split, X, y, chunk)
                for chunk in feature_chunks if len(chunk) > 0
            ]
            
            for future in future_results:
                results = future.result()
                for feature_idx, gain, threshold in results:
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_idx
                        best_threshold = threshold
        
        # If no good split is found, create leaf node
        if best_gain == -1:
            return Node(value=Counter(y).most_common(1)[0][0])
        
        # Create split node
        if self.feature_types[best_feature] == 'numerical':
            left_mask = X[:, best_feature] <= best_threshold
        else:
            left_mask = X[:, best_feature] == best_threshold
            
        right_mask = ~left_mask
        
        # Recursively grow subtrees
        left_subtree = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class for X using parallel processing"""
        X = np.array(X)
        
        def predict_batch(batch):
            return np.array([self._traverse_tree(x, self.root) for x in batch])
        
        # Split data into batches for parallel prediction
        batch_size = max(1, len(X) // self.n_jobs)
        batches = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            predictions = list(executor.map(predict_batch, batches))
            
        return np.concatenate(predictions)

    def _traverse_tree(self, x: np.ndarray, node: Node) -> int:
        """Helper method to traverse the tree"""
        if node.value is not None:
            return node.value

        if self.feature_types[node.feature] == 'numerical':
            if x[node.feature] <= node.threshold:
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)
        else:
            if x[node.feature] == node.threshold:
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)