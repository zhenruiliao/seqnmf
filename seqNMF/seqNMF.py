import numpy as np
import seaborn as sns
from scipy.signal import convolve2d as conv2
from matplotlib import pyplot as plt
from helpers import reconstruct, shift_factors, compute_loadings_percent_power

def seq_nmf(X, K=10, L=20, Lambda=.1, W_init=None, H_init=None, \
            plot_it=True, max_iter=20, tol=-np.inf, shift=True, sort_factors=True, \
            lambda_L1W=0, lambda_L1H=0, lambda_OrthH=0, lambda_OrthW=0, M=None, \
            use_W_update=True, W_fixed=False):

    N = X.shape[0]
    T = X.shape[1]

    if W_init is None:
        W_init = np.max(X) * np.random.rand(N, K, L)
    if H_init is None:
        H_init = np.max(X) * np.random.rand(K, T) / np.sqrt(T / 3)
    if M is None:
        M = np.ones([N, T])

    assert np.all(X >= 0), 'all data values must be positive!'

    W = W_init
    H = H_init

    X_hat = reconstruct(W, H)
    mask = M == 0
    X[mask] = X_hat[mask]

    smooth_kernel = np.ones([1, (2 * L) - 1])
    eps = np.max(X) * 1e-6
    last_time = False

    cost = np.zeros([max_iter + 1, 1])
    cost[0] = np.sqrt(np.mean(np.power(X - X_hat, 2)))

    for i in np.arange(max_iter):
        if (i == max_iter) or ((i > 6) and (cost[i + 1] + tol) > np.mean(cost[i - 6:i])):
            cost = cost[:i]
            last_time = True
            if i > 0:
                Lambda = 0

        WTX = np.zeros([K, T])
        WTX_hat = np.zeros([K, T])
        for j in np.arange(L):
            X_shifted = np.roll(X, -j + 1, axis=1)
            X_hat_shifted = np.roll(X_hat, -j + 1, axis=1)

            WTX += np.dot(W[:, :, j].T, X_shifted)
            WTX_hat += np.dot(W[:, :, j].T, X_hat_shifted)

        if Lambda > 0:
            dRdH = np.dot(Lambda * (1 - np.eye(K)), conv2(WTX, smooth_kernel, 'same'))
        else:
            dRdH = 0

        if lambda_OrthH > 0:
            dHHdH = np.dot(lambda_OrthH * (1 - np.eye(K)), conv2(H, smooth_kernel, 'same'))
        else:
            dHHdH = 0

        dRdH += lambda_L1H + dHHdH

        H *= np.divide(WTX, WTX_hat + dRdH + eps)

        if shift:
            W, H = shift_factors(W, H)
            W += eps

        norms = np.sqrt(np.sum(np.power(H, 2), axis=1)).T
        H = np.dot(np.diag(np.divide(1., norms + eps)), H)
        for j in np.arange(L):
            W[:, :, j] = np.dot(W[:, :, j], np.diag(norms))

        if not W_fixed:
            X_hat = reconstruct(W, H)
            mask = M == 0
            X[mask] = X_hat[mask]

            if lambda_OrthW > 0:
                W_flat = np.sum(W, axis=2)
            if (Lambda > 0) and use_W_update:
                XS = conv2(X, smooth_kernel, 'same')

            for j in np.arange(L):
                H_shifted = np.roll(H, j - 1, axis=1)
                XHT = np.dot(X, H_shifted.T)
                X_hat_HT = np.dot(X_hat, H_shifted.T)

                if (Lambda > 0) and use_W_update:
                    dRdW = Lambda * np.dot(np.dot(XS, H_shifted.T), (1. - np.eye(K)))
                else:
                    dRdW = 0

                if lambda_OrthW > 0:
                    dWWdW = np.dot(lambda_OrthW * W_flat, 1. - np.eye(K)) #TODO: CHECK THIS...
                else:
                    dWWdW = 0

                dRdW += lambda_L1W + dWWdW
                W[:, :, j] *= np.divide(XHT, X_hat_HT + dRdW + eps)

        X_hat = reconstruct(W, H)
        mask = M == 0
        X[mask] = X_hat[mask]
        cost[i] = np.sqrt(np.mean(np.power(X - X_hat, 2)))

        if plot_it:
            h = plot(W, H, X_hat)
            h.set_title(f'iteration {i}')

        if last_time:
            break

    #TODO: STOPPED DEBUGGING HERE
    X = X[:, L:-L]
    X_hat = X_hat[:, L:-L]
    H = H[:, L:-L]

    power = np.divide(np.sum(np.power(X, 2)) - np.sum(np.power(X - X_hat, 2)), np.sum(np.power(X, 2)))

    loadings = compute_loadings_percent_power(X, W, H)

    if sort_factors:
        inds = np.flip(np.argsort(loadings), 0)
        loadings = loadings[inds]

        W = W[:, inds, :]
        H = H[inds, :]

    return W, H, cost, loadings, power

def plot(W, H, cmap='Spectral'):
    data_recon = reconstruct(W, H)
    clims = [0, np.percentile(data_recon, 99)]

    f, (ax, ax_w, ax_h, ax_data) = plt.subplots(2, 2)

    sns.heatmap(W, cmap=cmap, ax=ax_w, vmin=clims[0], vmax=clims[1])
    sns.heatmap(H, cmap=cmap, ax=ax_h, vmin=clims[0], vmax=clims[1])
    sns.heatmap(data_recon, cmap=cmap, ax=ax_data, vmin=clims[0], vmax=clims[1])

    return f

