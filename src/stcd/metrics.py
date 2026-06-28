"""Denoising metrics.

Given per-event ground-truth labels (True = signal, False = noise) and a per-event
keep mask, we report the proposal's evaluation axes:

* **Signal Retain (SR)** — fraction of signal events kept (recall).
* **Noise Removal (NR)** — fraction of noise events dropped.
* **Denoise Accuracy (DA)** — overall fraction classified correctly.
* **SNR** in/out (dB) and the gain.
* **retention / reduction** — sparsity reported *jointly* with retention, because
  a filter that drops everything is trivially "sparse".
* **ROC / AUC** — parameter-free comparison by sweeping a per-event score.

``rpmd`` supports the DVSNOISE20 soft-plausibility (EPM) evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class DenoiseMetrics:
    signal_retain: float      # SR  (TP / signal)
    noise_removal: float      # NR  (TN / noise)
    denoise_accuracy: float   # DA  ((TP+TN)/N)
    f1: float
    snr_in_db: float
    snr_out_db: float
    snr_gain_db: float
    retention: float          # events kept / events in   (= 1 - reduction)
    reduction: float          # events dropped / events in
    n_in: int
    n_out: int
    n_signal: int
    n_noise: int

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"SR={self.signal_retain:.3f}  NR={self.noise_removal:.3f}  "
            f"DA={self.denoise_accuracy:.3f}  F1={self.f1:.3f}  "
            f"SNR {self.snr_in_db:+.1f}->{self.snr_out_db:+.1f} dB "
            f"(gain {self.snr_gain_db:+.1f})  "
            f"retain={self.retention:.3f} ({self.n_out}/{self.n_in})"
        )


def _snr_db(n_signal: int, n_noise: int) -> float:
    if n_noise == 0:
        return float("inf") if n_signal > 0 else 0.0
    return 10.0 * np.log10(max(n_signal, 1e-9) / n_noise)


def evaluate_filter(labels: np.ndarray, kept: np.ndarray) -> DenoiseMetrics:
    """Compute denoising metrics from boolean ground-truth ``labels`` (signal=True)
    and a boolean ``kept`` mask, both aligned per event."""
    labels = np.asarray(labels, dtype=bool)
    kept = np.asarray(kept, dtype=bool)
    if labels.shape != kept.shape:
        raise ValueError("labels and kept must align per event")

    n = len(labels)
    n_signal = int(labels.sum())
    n_noise = n - n_signal

    tp = int((kept & labels).sum())
    fn = n_signal - tp
    fp = int((kept & ~labels).sum())
    tn = n_noise - fp
    n_out = tp + fp

    sr = tp / n_signal if n_signal else 0.0
    nr = tn / n_noise if n_noise else 1.0
    da = (tp + tn) / n if n else 0.0
    precision = tp / n_out if n_out else 0.0
    f1 = (2 * precision * sr / (precision + sr)) if (precision + sr) else 0.0

    snr_in = _snr_db(n_signal, n_noise)
    snr_out = _snr_db(tp, fp)
    gain = (snr_out - snr_in) if np.isfinite(snr_out) and np.isfinite(snr_in) else float("inf")

    return DenoiseMetrics(
        signal_retain=sr,
        noise_removal=nr,
        denoise_accuracy=da,
        f1=f1,
        snr_in_db=snr_in,
        snr_out_db=snr_out,
        snr_gain_db=gain,
        retention=n_out / n if n else 0.0,
        reduction=1.0 - (n_out / n if n else 0.0),
        n_in=n,
        n_out=n_out,
        n_signal=n_signal,
        n_noise=n_noise,
    )


def roc(scores: np.ndarray, labels: np.ndarray) -> dict:
    """ROC of a per-event signal-support ``score`` against ``labels`` (signal=positive).

    Returns ``fpr`` (= fraction of noise kept = 1−NR), ``tpr`` (= SR), the
    thresholds, and the AUC. Uses scikit-learn when available, else a NumPy
    fallback.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    try:
        from sklearn.metrics import roc_curve, roc_auc_score

        fpr, tpr, thr = roc_curve(labels, scores)
        auc = float(roc_auc_score(labels, scores)) if labels.any() and (~labels).any() else float("nan")
        return {"fpr": fpr, "tpr": tpr, "thresholds": thr, "auc": auc}
    except Exception:
        return _roc_numpy(scores, labels)


def _roc_numpy(scores: np.ndarray, labels: np.ndarray) -> dict:
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order].astype(np.int64)
    P, N = int(y.sum()), int((1 - y).sum())
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = np.concatenate([[0.0], tp / max(P, 1)])
    fpr = np.concatenate([[0.0], fp / max(N, 1)])
    auc = float(np.trapz(tpr, fpr)) if N and P else float("nan")
    return {"fpr": fpr, "tpr": tpr, "thresholds": np.concatenate([[np.inf], s]), "auc": auc}


def rpmd(plausibility: np.ndarray, kept: np.ndarray) -> float:
    """Relative Plausibility Mean Difference (DVSNOISE20 / EDnCNN style).

    ``plausibility`` in [0,1] is the EPM probability that each event is real.
    Score = mean plausibility of *kept* events − mean plausibility of *removed*
    events, scaled to a "lower is better" distance: ``1 − (kept_mean − removed_mean)``.
    A perfect denoiser keeps all plausible events and drops implausible ones,
    maximising the gap. Requires the dataset's soft EPM labels.
    """
    plausibility = np.asarray(plausibility, dtype=np.float64)
    kept = np.asarray(kept, dtype=bool)
    if kept.all() or (~kept).any() is False:
        return float("nan")
    kept_mean = plausibility[kept].mean() if kept.any() else 0.0
    removed_mean = plausibility[~kept].mean() if (~kept).any() else 0.0
    return float(1.0 - (kept_mean - removed_mean))


def sweep_threshold(scores: np.ndarray, labels: np.ndarray,
                    thresholds: np.ndarray) -> list[DenoiseMetrics]:
    """Evaluate the denoiser at each score threshold (keep iff score ≥ thr)."""
    return [evaluate_filter(labels, scores >= thr) for thr in thresholds]


def kind_subset_metrics(
    labels: np.ndarray, kept: np.ndarray, kinds: np.ndarray,
    noise_kinds, scores: np.ndarray | None = None, signal_kind: int = 0,
) -> dict:
    """Evaluate a denoiser against ONE noise process in isolation.

    Restricts the event set to *signal* events (``kinds == signal_kind``) plus the
    events whose ``kinds`` is in ``noise_kinds``, then reports the usual
    :func:`evaluate_filter` metrics (SR / NR / DA / F1 / SNR / retention) and, if a
    per-event ``scores`` array is given, the ROC-AUC on that subset. This is what
    lets the Idea-1 bake-off report 1a (hot pixels) and 1b (correlated readout)
    separately — an aggregate metric hides which process a strategy actually beats
    (e.g. the neighbour-support multiple #3 is expected to fail on 1b).
    """
    labels = np.asarray(labels, dtype=bool)
    kept = np.asarray(kept, dtype=bool)
    kinds = np.asarray(kinds, dtype=np.int64)
    noise_kinds = [int(k) for k in np.atleast_1d(noise_kinds)]
    sub = (kinds == signal_kind) | np.isin(kinds, noise_kinds)
    out = evaluate_filter(labels[sub], kept[sub]).as_dict()
    if scores is not None:
        scores = np.asarray(scores, dtype=np.float64)
        out["auc"] = float(roc(scores[sub], labels[sub])["auc"])
    out["n_sub"] = int(sub.sum())
    return out
