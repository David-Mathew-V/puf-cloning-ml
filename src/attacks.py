"""ML attacks on arbiter PUFs: perceptron, LR, SVMs, MLP and product-LR."""
import time

import numpy as np
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC


class Attack:
    def __init__(self, name, model):
        self.name, self.model = name, model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


def perceptron(seed=0):
    return Attack("Perceptron", Perceptron(max_iter=200, tol=1e-4, random_state=seed))


def logistic_regression(C=10.0):
    return Attack("Logistic regression", LogisticRegression(C=C, max_iter=5000))


def linear_svm(C=1.0):
    return Attack("Linear SVM", LinearSVC(C=C, max_iter=20000))


def rbf_svm(C=10.0):
    return Attack("RBF-kernel SVM", SVC(C=C, kernel="rbf", gamma="scale", cache_size=2000))


def mlp(k, N, seed=0):
    h = max(8, 2 ** k)
    model = MLPClassifier(
        hidden_layer_sizes=(h, h, h), activation="tanh", solver="adam",
        alpha=1e-5, learning_rate_init=3e-3, batch_size=int(min(4096, max(32, N // 50))),
        max_iter=400, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=12, tol=1e-5, random_state=seed)
    return Attack("MLP", model)


class ProductLR:
    """P(y=1) = sigmoid(a * prod_k (w_k . phi)), trained with Adam + random restarts."""

    name = "Product-LR"

    def __init__(self, k, lr=0.01, max_epochs=300, patience=10,
                 restarts=6, val_frac=0.1, seed=0, target_acc=0.995):
        self.k, self.lr, self.max_epochs, self.patience = k, lr, max_epochs, patience
        self.restarts, self.val_frac, self.seed, self.target_acc = restarts, val_frac, seed, target_acc
        self.W, self.a = None, 1.0

    @staticmethod
    def _scores(X, W, a):
        return a * np.prod(X @ W.T, axis=1)

    def _acc(self, X, y, W, a):
        return float(np.mean((self._scores(X, W, a) > 0).astype(np.int64) == y))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(y))
        n_val = int(min(max(200, self.val_frac * len(y)), 20000))
        va, tr = idx[:n_val], idx[n_val:]
        Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
        d = X.shape[1]
        bs = int(min(4096, max(32, len(tr) // 50)))
        best = (-1.0, None, None)

        for r in range(self.restarts):
            W = rng.normal(0, 1 / np.sqrt(d), size=(self.k, d))
            a = 1.0
            mW, vW, ma, va_ = np.zeros_like(W), np.zeros_like(W), 0.0, 0.0
            b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
            run_best, run_W, run_a, bad = -1.0, W.copy(), a, 0

            for _ in range(self.max_epochs):
                perm = rng.permutation(len(ytr))
                for i in range(0, len(ytr), bs):
                    b = perm[i:i + bs]
                    Xb, yb = Xtr[b], ytr[b]
                    Z = Xb @ W.T
                    P = np.prod(Z, axis=1)
                    p = 0.5 * (1 + np.tanh(0.5 * a * P))  # sigmoid
                    g = (p - yb) / len(yb)
                    ga = float(np.sum(g * P))
                    gW = np.empty_like(W)
                    for j in range(self.k):
                        others = np.prod(np.delete(Z, j, axis=1), axis=1) if self.k > 1 else 1.0
                        gW[j] = (g * a * others) @ Xb
                    # adam
                    t += 1
                    mW = b1 * mW + (1 - b1) * gW; vW = b2 * vW + (1 - b2) * gW ** 2
                    ma = b1 * ma + (1 - b1) * ga; va_ = b2 * va_ + (1 - b2) * ga ** 2
                    W -= self.lr * (mW / (1 - b1 ** t)) / (np.sqrt(vW / (1 - b2 ** t)) + eps)
                    a -= self.lr * (ma / (1 - b1 ** t)) / (np.sqrt(va_ / (1 - b2 ** t)) + eps)

                acc = self._acc(Xva, yva, W, a)
                if acc > run_best + 1e-4:
                    run_best, run_W, run_a, bad = acc, W.copy(), a, 0
                else:
                    bad += 1
                if bad >= self.patience or run_best >= self.target_acc:
                    break

            if run_best > best[0]:
                best = (run_best, run_W, run_a)
            if best[0] >= self.target_acc:
                break

        self.val_acc, self.W, self.a = best
        return self

    def predict(self, X):
        return (self._scores(np.asarray(X, dtype=np.float64), self.W, self.a) > 0).astype(np.int64)


def timed_fit_eval(attack, Xtr, ytr, Xte, yte):
    t0 = time.perf_counter()
    attack.fit(Xtr, ytr)
    t_fit = time.perf_counter() - t0
    acc = float(np.mean(attack.predict(Xte) == yte))
    return acc, t_fit
