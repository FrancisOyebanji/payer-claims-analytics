"""Multivariate Adaptive Regression Splines (MARS) — from-scratch implementation.

Implements the additive (degree-1) form of Friedman (1991):
  1. Forward pass: greedily add mirrored hinge pairs max(0, x-t) / max(0, t-x),
     choosing the feature/knot that most reduces squared error.
  2. Backward pass: prune basis functions using generalized cross-validation (GCV).

For the binary high-cost target, basis selection runs on squared error against
the 0/1 response (the classic MARS-for-classification recipe), and a logistic
regression is then fit on the selected basis to produce calibrated probabilities.

Why from scratch: the reference package (py-earth) is unmaintained and no longer
builds against modern numpy/scikit-learn. This implementation keeps the method
available, sklearn-compatible, and small enough to review line by line.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class _Hinge:
    __slots__ = ("feature", "knot", "direction")

    def __init__(self, feature: int, knot: float, direction: int):
        self.feature = feature      # column index
        self.knot = knot            # threshold t
        self.direction = direction  # +1 -> max(0, x - t); -1 -> max(0, t - x)

    def transform(self, X: np.ndarray) -> np.ndarray:
        z = (X[:, self.feature] - self.knot) * self.direction
        return np.maximum(z, 0.0)

    def __repr__(self) -> str:
        return (f"h(x{self.feature} - {self.knot:.3g})" if self.direction == 1
                else f"h({self.knot:.3g} - x{self.feature})")


class MARSClassifier(BaseEstimator, ClassifierMixin):
    """Additive MARS with logistic output layer.

    Parameters
    ----------
    max_terms : forward-pass budget of hinge basis functions (pairs count as 2).
    n_knots   : candidate knots per feature (quantile grid).
    penalty   : GCV cost per basis function (Friedman recommends 2-3 for additive).
    """

    def __init__(self, max_terms: int = 24, n_knots: int = 7, penalty: float = 3.0):
        self.max_terms = max_terms
        self.n_knots = n_knots
        self.penalty = penalty

    # ---------- internals ----------
    @staticmethod
    def _lstsq_sse(B: np.ndarray, y: np.ndarray) -> float:
        coef, _, _, _ = np.linalg.lstsq(B, y, rcond=None)
        resid = y - B @ coef
        return float(resid @ resid)

    def _gcv(self, sse: float, n: int, n_basis: int) -> float:
        # effective params: basis fns + penalty per selected knot
        c = n_basis + self.penalty * (n_basis - 1) / 2
        denom = (1 - c / n) ** 2
        return np.inf if denom <= 0 else sse / (n * denom)

    def _basis_matrix(self, X: np.ndarray, hinges: list[_Hinge]) -> np.ndarray:
        cols = [np.ones(len(X))] + [h.transform(X) for h in hinges]
        return np.column_stack(cols)

    # ---------- fitting ----------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)

        # Candidate knots: interior quantiles per feature (skip constant features)
        qs = np.linspace(0.1, 0.9, self.n_knots)
        candidates: list[tuple[int, float]] = []
        for j in range(X.shape[1]):
            knots = np.unique(np.quantile(X[:, j], qs))
            if len(knots) > 1:
                candidates.extend((j, float(t)) for t in knots)

        # --- Forward pass: add best mirrored hinge pair each iteration ---
        hinges: list[_Hinge] = []
        B = np.ones((n, 1))
        current_sse = self._lstsq_sse(B, y)
        while len(hinges) < self.max_terms:
            best = None
            for j, t in candidates:
                pair = [_Hinge(j, t, +1), _Hinge(j, t, -1)]
                trial = np.column_stack([B] + [h.transform(X) for h in pair])
                sse = self._lstsq_sse(trial, y)
                if best is None or sse < best[0]:
                    best = (sse, pair)
            if best is None or best[0] >= current_sse * (1 - 1e-4):
                break  # no meaningful improvement
            current_sse = best[0]
            hinges.extend(best[1])
            B = np.column_stack([B] + [h.transform(X) for h in best[1]])
            # a knot may be reused only via remaining candidates; drop the used one
            used = (best[1][0].feature, best[1][0].knot)
            candidates = [c for c in candidates if c != used]

        # --- Backward pass: prune by GCV ---
        best_set = list(hinges)
        best_gcv = self._gcv(current_sse, n, len(hinges) + 1)
        working = list(hinges)
        while working:
            scored = []
            for i in range(len(working)):
                subset = working[:i] + working[i + 1:]
                sse = self._lstsq_sse(self._basis_matrix(X, subset), y)
                scored.append((self._gcv(sse, n, len(subset) + 1), subset))
            gcv, subset = min(scored, key=lambda s: s[0])
            working = subset
            if gcv < best_gcv:
                best_gcv, best_set = gcv, list(subset)

        self.hinges_ = best_set
        self.gcv_ = best_gcv

        # --- Logistic output layer on the selected basis ---
        self._logit = LogisticRegression(max_iter=2000)
        self._logit.fit(self._basis_matrix(X, self.hinges_)[:, 1:], y.astype(int))
        self.classes_ = self._logit.classes_
        return self

    # ---------- inference ----------
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return self._logit.predict_proba(self._basis_matrix(X, self.hinges_)[:, 1:])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def describe(self) -> list[str]:
        """Human-readable selected basis functions (for model review)."""
        return [repr(h) for h in self.hinges_]
