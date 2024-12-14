import numpy as np

class GaussianNB():
    """
    Ref: https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf
    Uses estimate MLE for mean and MVUE for variance
    """

    def __init__(self) :
        pass

    def fit(self, X, y):
        self.classes_, class_count_ = np.unique(y, return_counts=True)
        
        # ndarray of shape (n_classes,)
        self.class_priors_ = class_count_ / y.shape[0]

        # ndarray of shape (n_classes, n_features)
        self.theta_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        self.var_ = np.array([X[y == c].var(axis=0) for c in self.classes_])

        self.var_ = np.maximum(self.var_, 1e-9)

    def predict(self, X):
        log_priors = np.log(self.class_priors_)

        log_likelihoods = []
        for i, _ in enumerate(self.classes_):
            # likelihood: feature xi given class Ck
            # X - self.theta_[i] involves broadcasting self.theta_[i] to each row of X
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.var_[i])) - 0.5 * np.sum(((X - self.theta_[i]) ** 2) / self.var_[i], axis=1)
            log_likelihoods.append(log_likelihood)

        log_likelihoods = np.array(log_likelihoods).T
        log_posteriors = log_likelihoods + log_priors

        return self.classes_[np.argmax(log_posteriors, axis=1)]
