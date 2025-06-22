import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression

from src.ml_algorithms import (
    LinearRegression,
    KMeans,
    NaiveBayes,
    train_test_split,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
)


class TestLinearRegression:
    def test_initialization(self):
        lr = LinearRegression()
        assert lr.learning_rate == 0.01
        assert lr.max_iterations == 1000
        assert lr.weights is None
        assert lr.bias is None

    def test_custom_parameters(self):
        lr = LinearRegression(learning_rate=0.05, max_iterations=500)
        assert lr.learning_rate == 0.05
        assert lr.max_iterations == 500

    def test_fit_and_predict(self):
        # Create simple dataset
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])  # y = 2*x

        lr = LinearRegression(learning_rate=0.1, max_iterations=1000)
        lr.fit(X, y)

        # Check that weights and bias are learned
        assert lr.weights is not None
        assert lr.bias is not None
        assert len(lr.cost_history) == 1000

        # Test prediction
        predictions = lr.predict(X)
        assert len(predictions) == len(y)

        # Check that the model learns the correct relationship (approximately)
        mse = mean_squared_error(y, predictions)
        assert mse < 1.0  # Should be quite accurate for this simple case

    def test_multivariate_regression(self):
        X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)

        lr = LinearRegression(learning_rate=0.01, max_iterations=1000)
        lr.fit(X, y)

        predictions = lr.predict(X)
        mse = mean_squared_error(y, predictions)

        # Should achieve reasonable accuracy
        assert mse < 10.0
        assert lr.weights.shape == (3,)


class TestKMeans:
    def test_initialization(self):
        kmeans = KMeans()
        assert kmeans.k == 3
        assert kmeans.max_iterations == 100
        assert kmeans.centroids is None
        assert kmeans.labels is None

    def test_custom_parameters(self):
        kmeans = KMeans(k=5, max_iterations=50, random_state=42)
        assert kmeans.k == 5
        assert kmeans.max_iterations == 50
        assert kmeans.random_state == 42

    def test_fit_and_predict(self):
        # Create simple 2D dataset with clear clusters
        np.random.seed(42)
        cluster1 = np.random.normal([0, 0], 0.5, (20, 2))
        cluster2 = np.random.normal([3, 3], 0.5, (20, 2))
        cluster3 = np.random.normal([0, 3], 0.5, (20, 2))
        X = np.vstack([cluster1, cluster2, cluster3])

        kmeans = KMeans(k=3, random_state=42)
        kmeans.fit(X)

        # Check that centroids and labels are assigned
        assert kmeans.centroids is not None
        assert kmeans.labels is not None
        assert kmeans.centroids.shape == (3, 2)
        assert len(kmeans.labels) == 60

        # Test prediction on new data
        new_points = np.array([[0, 0], [3, 3]])
        predictions = kmeans.predict(new_points)
        assert len(predictions) == 2
        assert all(0 <= pred < 3 for pred in predictions)

    def test_convergence(self):
        # Test with well-separated clusters
        np.random.seed(42)
        X = np.array([[1, 1], [1, 2], [2, 1], [8, 8], [8, 9], [9, 8]])

        kmeans = KMeans(k=2, random_state=42)
        kmeans.fit(X)

        # Should converge to two clear clusters
        assert kmeans.centroids.shape == (2, 2)
        # Check that points are assigned to correct clusters
        labels = kmeans.labels
        # Points 0,1,2 should be in one cluster, points 3,4,5 in another
        assert len(set(labels[:3])) == 1 or len(set(labels[3:])) == 1


class TestNaiveBayes:
    def test_initialization(self):
        nb = NaiveBayes()
        assert nb.class_priors == {}
        assert nb.feature_probs == {}
        assert nb.classes is None

    def test_fit_and_predict(self):
        # Create simple classification dataset
        X, y = make_classification(
            n_samples=100, n_features=4, n_classes=2, random_state=42
        )

        nb = NaiveBayes()
        nb.fit(X, y)

        # Check that model parameters are learned
        assert len(nb.class_priors) == 2
        assert len(nb.feature_probs) == 2
        assert nb.classes is not None
        assert len(nb.classes) == 2

        # Test prediction
        predictions = nb.predict(X)
        assert len(predictions) == len(y)
        assert all(pred in nb.classes for pred in predictions)

        # Check prediction probabilities
        probabilities = nb.predict_proba(X)
        assert probabilities.shape == (100, 2)
        # Probabilities should sum to approximately 1
        prob_sums = np.sum(probabilities, axis=1)
        assert np.allclose(prob_sums, 1.0, atol=1e-6)

    def test_accuracy(self):
        X, y = make_classification(
            n_samples=200,
            n_features=4,
            n_classes=2,
            n_informative=3,
            n_redundant=1,
            random_state=42,
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Train model
        nb = NaiveBayes()
        nb.fit(X_train, y_train)

        # Make predictions
        predictions = nb.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        # Should achieve reasonable accuracy
        assert acc > 0.7


class TestUtilityFunctions:
    def test_train_test_split(self):
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        assert X_train.shape == (80, 5)
        assert X_test.shape == (20, 5)
        assert len(y_train) == 80
        assert len(y_test) == 20

        # Check that no data is duplicated
        assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
        assert len(y_train) + len(y_test) == len(y)

    def test_accuracy_score(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        acc = accuracy_score(y_true, y_pred)
        expected_acc = 4 / 5  # 4 correct out of 5

        assert acc == expected_acc

        # Test perfect accuracy
        acc_perfect = accuracy_score(y_true, y_true)
        assert acc_perfect == 1.0

    def test_mean_squared_error(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        mse = mean_squared_error(y_true, y_pred)
        assert mse == 0.0

        # Test with known values
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])
        mse = mean_squared_error(y_true, y_pred)
        assert mse == 1.0  # (1² + 1² + 1²) / 3 = 1

    def test_mean_absolute_error(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        mae = mean_absolute_error(y_true, y_pred)
        assert mae == 0.0

        # Test with known values
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])
        mae = mean_absolute_error(y_true, y_pred)
        assert mae == 1.0  # (1 + 1 + 1) / 3 = 1


@pytest.mark.slow
class TestIntegrationTests:
    def test_full_ml_pipeline(self):
        # Create dataset
        X, y = make_classification(
            n_samples=500, n_features=10, n_classes=3, n_informative=8, random_state=42
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train Naive Bayes
        nb = NaiveBayes()
        nb.fit(X_train, y_train)
        nb_pred = nb.predict(X_test)
        nb_acc = accuracy_score(y_test, nb_pred)

        # Train KMeans (unsupervised)
        kmeans = KMeans(k=3, random_state=42)
        kmeans.fit(X_train)
        cluster_pred = kmeans.predict(X_test)

        # Basic checks
        assert nb_acc > 0.5  # Should be better than random
        assert len(cluster_pred) == len(X_test)
        assert all(0 <= pred < 3 for pred in cluster_pred)
