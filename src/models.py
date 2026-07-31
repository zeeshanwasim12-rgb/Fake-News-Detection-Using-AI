"""
Week 3 - Model Building (all algorithms implemented from scratch with NumPy)
------------------------------------------------------------------------
  1. KNN               - Non-parametric
  2. Logistic Regression - Parametric (batch gradient descent)
  3. Random Forest      - Ensemble of from-scratch Decision Trees (bagging)
  4. Simple Neural Net  - one hidden layer, manual forward/backprop

No sklearn.neighbors / linear_model / ensemble / neural_network is used.
"""

import numpy as np


# ----------------------------------------------------------------------
# 1. K-Nearest Neighbors
# ----------------------------------------------------------------------
class KNNScratch:
    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        preds = np.empty(X.shape[0], dtype=self.y_train.dtype)
        # Vectorized squared-euclidean distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        train_sq = np.sum(self.X_train ** 2, axis=1)
        for i in range(X.shape[0]):
            x = X[i]
            dists = train_sq + np.sum(x ** 2) - 2 * self.X_train.dot(x)
            nn_idx = np.argpartition(dists, min(self.k, len(dists) - 1))[: self.k]
            nn_labels = self.y_train[nn_idx]
            vals, counts = np.unique(nn_labels, return_counts=True)
            preds[i] = vals[np.argmax(counts)]
        return preds


# ----------------------------------------------------------------------
# 2. Logistic Regression (parametric, gradient descent)
# ----------------------------------------------------------------------
class LogisticRegressionScratch:
    def __init__(self, lr: float = 0.5, n_iters: int = 300, l2: float = 0.001):
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).astype(np.float64)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iters):
            z = X.dot(self.w) + self.b
            p = self._sigmoid(z)
            grad_w = X.T.dot(p - y) / n + self.l2 * self.w
            grad_b = np.mean(p - y)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        return self._sigmoid(np.asarray(X).dot(self.w) + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# ----------------------------------------------------------------------
# 3. Decision Tree + Random Forest (ensemble, bagging)
# ----------------------------------------------------------------------
class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeScratch:
    def __init__(self, max_depth: int = 8, min_samples_split: int = 4, n_feat_sample: int = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_feat_sample = n_feat_sample  # for random-forest feature subsampling

    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p ** 2)

    def _best_split(self, X, y, feat_idxs):
        best_gain, best_feat, best_thr = -1, None, None
        parent_gini = self._gini(y)
        n = len(y)
        for feat in feat_idxs:
            col = X[:, feat]
            thresholds = np.unique(col)
            if len(thresholds) > 10:  # subsample thresholds for speed
                thresholds = np.quantile(col, np.linspace(0.1, 0.9, 9))
            for thr in thresholds:
                left_mask = col <= thr
                n_left, n_right = left_mask.sum(), n - left_mask.sum()
                if n_left == 0 or n_right == 0:
                    continue
                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[~left_mask])
                weighted = (n_left / n) * gini_left + (n_right / n) * gini_right
                gain = parent_gini - weighted
                if gain > best_gain:
                    best_gain, best_feat, best_thr = gain, feat, thr
        return best_feat, best_thr, best_gain

    def _build(self, X, y, depth):
        if (depth >= self.max_depth or len(y) < self.min_samples_split or
                len(np.unique(y)) == 1):
            return _Node(value=np.bincount(y, minlength=2).argmax())

        n_features = X.shape[1]
        if self.n_feat_sample:
            feat_idxs = np.random.choice(n_features, self.n_feat_sample, replace=False)
        else:
            feat_idxs = np.arange(n_features)

        feat, thr, gain = self._best_split(X, y, feat_idxs)
        if feat is None or gain <= 0:
            return _Node(value=np.bincount(y, minlength=2).argmax())

        left_mask = X[:, feat] <= thr
        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return _Node(feature=feat, threshold=thr, left=left, right=right)

    def fit(self, X, y):
        self.root = self._build(np.asarray(X), np.asarray(y), 0)
        return self

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(x, self.root) for x in X])


class RandomForestScratch:
    def __init__(self, n_trees: int = 15, max_depth: int = 8, sample_ratio: float = 0.8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_ratio = sample_ratio
        self.trees = []

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        n, d = X.shape
        n_feat_sample = max(1, int(np.sqrt(d)))
        n_sample = int(n * self.sample_ratio)
        self.trees = []
        for _ in range(self.n_trees):
            idxs = np.random.choice(n, n_sample, replace=True)
            tree = DecisionTreeScratch(max_depth=self.max_depth, n_feat_sample=n_feat_sample)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)
        return self

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])  # (n_trees, n_samples)
        # majority vote
        return np.apply_along_axis(lambda col: np.bincount(col).argmax(), axis=0, arr=preds)


# ----------------------------------------------------------------------
# 4. Simple Neural Network (1 hidden layer, manual backprop)
# ----------------------------------------------------------------------
class SimpleNeuralNetScratch:
    def __init__(self, hidden_size: int = 32, lr: float = 0.05, n_iters: int = 300, l2: float = 1e-4):
        self.hidden_size = hidden_size
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2

    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    @staticmethod
    def _relu_deriv(z):
        return (z > 0).astype(z.dtype)

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).astype(np.float64).reshape(-1, 1)
        n, d = X.shape
        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0, np.sqrt(2 / d), size=(d, self.hidden_size))
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = rng.normal(0, np.sqrt(2 / self.hidden_size), size=(self.hidden_size, 1))
        self.b2 = np.zeros((1, 1))

        for _ in range(self.n_iters):
            # forward
            Z1 = X.dot(self.W1) + self.b1
            A1 = self._relu(Z1)
            Z2 = A1.dot(self.W2) + self.b2
            A2 = self._sigmoid(Z2)

            # backward (binary cross-entropy)
            dZ2 = (A2 - y) / n
            dW2 = A1.T.dot(dZ2) + self.l2 * self.W2
            db2 = dZ2.sum(axis=0, keepdims=True)
            dA1 = dZ2.dot(self.W2.T)
            dZ1 = dA1 * self._relu_deriv(Z1)
            dW1 = X.T.dot(dZ1) + self.l2 * self.W1
            db1 = dZ1.sum(axis=0, keepdims=True)

            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        A1 = self._relu(X.dot(self.W1) + self.b1)
        A2 = self._sigmoid(A1.dot(self.W2) + self.b2)
        return A2.ravel()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
