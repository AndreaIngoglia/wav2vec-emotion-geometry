"""
common.geometry - linear-algebra helpers for comparing probe weight directions
and subspaces: row-wise cosine, principal angles (QR-based and scipy.orth+SVD)
and scalar subspace-overlap measures.
"""

from __future__ import annotations

import numpy as np


def l2_normalize_rows(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A = A.astype(np.float64, copy=False)
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return A / norms


def cosine_rowwise(W: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Row-wise cosine between each row of ``W`` (C, D) and a vector ``w`` (D,)."""
    if w.ndim == 2:
        w = w.reshape(-1)
    Wn = l2_normalize_rows(W)
    wn = w.astype(np.float64)
    wn = wn / (np.linalg.norm(wn) + 1e-12)
    return Wn @ wn


def angle_degrees_from_cos(cos_vals: np.ndarray) -> np.ndarray:
    cos_vals = np.clip(cos_vals, -1.0, 1.0)
    return np.degrees(np.arccos(cos_vals))


def row_center(W: np.ndarray) -> np.ndarray:
    """Subtract the mean row -- removes the offset shared across multiclass rows."""
    W = W.astype(np.float64, copy=False)
    return W - W.mean(axis=0, keepdims=True)


def orthonormal_basis_from_rows(A: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis ``(D, r)`` for the span of the rows of ``A``."""
    A = A.astype(np.float64, copy=False)
    if np.linalg.norm(A) < 1e-12:
        return np.zeros((A.shape[1], 0), dtype=np.float64)

    M = A.T
    Q, R = np.linalg.qr(M)
    if R.size == 0:
        v = A[0].copy()
        v /= (np.linalg.norm(v) + 1e-12)
        return v.reshape(-1, 1)

    diag = np.abs(np.diag(R))
    r = int(np.sum(diag > 1e-10))
    if r == 0:
        v = A[0].copy()
        v /= (np.linalg.norm(v) + 1e-12)
        return v.reshape(-1, 1)
    return Q[:, :r]


def principal_angles_cosines(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosines of the principal angles between span(A) and span(B) (descending)."""
    UA = orthonormal_basis_from_rows(A)
    UB = orthonormal_basis_from_rows(B)
    if UA.shape[1] == 0 or UB.shape[1] == 0:
        return np.array([], dtype=np.float64)
    s = np.linalg.svd(UA.T @ UB, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def principal_angles_degrees(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (degrees, ascending) between span(A) and span(B)."""
    cosines = principal_angles_cosines(A, B)
    if cosines.size == 0:
        return np.array([], dtype=np.float64)
    return np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))


def subspace_overlap_measures(A: np.ndarray, B: np.ndarray) -> dict:
    """Scalar overlap measures between span(A) and span(B)."""
    UA = orthonormal_basis_from_rows(A)
    UB = orthonormal_basis_from_rows(B)
    rA, rB = UA.shape[1], UB.shape[1]

    cosines = principal_angles_cosines(A, B)
    k = int(cosines.size)
    if k == 0:
        return {
            "rank_A": int(rA), "rank_B": int(rB), "k": 0,
            "smallest_angle_deg": None, "mean_cos2": None, "affinity": None,
        }

    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    denom = max(rA * rB, 1)
    return {
        "rank_A": int(rA),
        "rank_B": int(rB),
        "k": int(k),
        "smallest_angle_deg": float(angles[0]),
        "mean_cos2": float(np.mean(cosines ** 2)),
        "affinity": float(np.sqrt(np.sum(cosines ** 2) / denom)),
    }


def principal_angles_orth(W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """Principal angles (degrees, ascending) via ``scipy.linalg.orth`` + SVD.

    ``W1`` and ``W2`` are row-stacks of vectors; their row spans are compared.
    """
    from scipy.linalg import svd, orth

    U1 = orth(W1.T)
    U2 = orth(W2.T)
    M = U1.T @ U2
    _, sigmas, _ = svd(M, full_matrices=False)
    sigmas = np.clip(sigmas, 0, 1)
    return np.sort(np.degrees(np.arccos(sigmas)))
