import numpy as np
from typing import Tuple, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array


class LinearRegression:
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []

        for i in range(self.max_iterations):
            y_pred = self.predict(X)

            cost = np.mean((y - y_pred) ** 2) / 2
            self.cost_history.append(cost)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias


class KMeans:
    def __init__(
        self, k: int = 3, max_iterations: int = 100, random_state: Optional[int] = None
    ):
        self.k = k
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, X: np.ndarray) -> "KMeans":
        if self.random_state:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # Initialize centroids randomly
        self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]

        for _ in range(self.max_iterations):
            # Assign points to closest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)

            # Update centroids
            new_centroids = np.array(
                [X[self.labels == i].mean(axis=0) for i in range(self.k)]
            )

            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)


class NaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBayes":
        X, y = check_X_y(X, y)

        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        for class_val in self.classes:
            class_mask = y == class_val
            self.class_priors[class_val] = np.sum(class_mask) / n_samples

            X_class = X[class_mask]
            self.feature_probs[class_val] = {
                "mean": np.mean(X_class, axis=0),
                "var": np.var(X_class, axis=0),
            }

        return self

    def _gaussian_probability(self, x: float, mean: float, var: float) -> float:
        epsilon = 1e-6  # Prevent division by zero
        return (1 / np.sqrt(2 * np.pi * (var + epsilon))) * np.exp(
            -0.5 * ((x - mean) ** 2) / (var + epsilon)
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X)
        probabilities = []

        for sample in X:
            sample_probs = []
            for class_val in self.classes:
                class_prob = self.class_priors[class_val]

                feature_prob = 1.0
                for i, feature_val in enumerate(sample):
                    mean = self.feature_probs[class_val]["mean"][i]
                    var = self.feature_probs[class_val]["var"][i]
                    feature_prob *= self._gaussian_probability(feature_val, mean, var)

                sample_probs.append(class_prob * feature_prob)

            # Normalize probabilities
            total_prob = sum(sample_probs)
            if total_prob > 0:
                sample_probs = [p / total_prob for p in sample_probs]
            else:
                sample_probs = [1.0 / len(self.classes)] * len(self.classes)

            probabilities.append(sample_probs)

        return np.array(probabilities)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes[np.argmax(probabilities, axis=1)]


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if random_state:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)

    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true == y_pred)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))
