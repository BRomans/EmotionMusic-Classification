import numpy as np
from meegkit.asr import ASR
from meegkit.utils.matrix import sliding_window
from scipy import sparse
from scipy.sparse.linalg import spsolve

#  Resample the annotations and stretch them over 60 seconds depending on the amount of data points


def compute_asr_reconstruction(eeg, train_duration=10, train_baseline=None, sfreq=250, win_len=0.5, win_overlap=0.66):

    # If no baseline is provided, use the a portion of the eeg signal itself
    if train_baseline is None:
        train_baseline = eeg

    # Train on a clean portion of data
    asr = ASR(method='euclid' , win_len=win_len, win_overlap=win_overlap)
    train_idx = np.arange(0 * sfreq, train_duration * sfreq, dtype=int)
    _, sample_mask = asr.fit(train_baseline.matrix_data[:, train_idx])

    # Apply filter using sliding (non-overlapping) windows
    X = sliding_window(eeg.matrix_data, window=int(sfreq), step=int(sfreq))
    Y = np.zeros_like(X)
    for i in range(X.shape[1]):
        Y[:, i, :] = asr.transform(X[:, i, :])

    #raw_data = X.reshape(2, -1)  # reshape to (n_chans, n_times)
    clean_data = Y.reshape(2, -1)
    return clean_data


def baseline_als_optimized(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z