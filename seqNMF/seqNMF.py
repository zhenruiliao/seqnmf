import numpy as np
import seaborn as sns
from scipy.signal import convolve2d as conv2
from matplotlib import pyplot as plt
from matplotlib import gridspec
from helpers import reconstruct, shift_factors, compute_loadings_percent_power, get_shapes

def seq_nmf(X, K=10, L=100, Lambda=.001, W_init=None, H_init=None, \
            plot_it=True, max_iter=100, tol=-np.inf, shift=True, sort_factors=True, \
            lambda_L1W=0, lambda_L1H=0, lambda_OrthH=0, lambda_OrthW=0, M=None, \
            use_W_update=True, W_fixed=False):

    N = X.shape[0]
    T = X.shape[1] + 2*L
    X = np.concatenate((np.zeros([N, L]), X, np.zeros([N, L])), axis=1)

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
        if (i == max_iter-1) or ((i > 6) and (cost[i + 1] + tol) > np.mean(cost[i - 6:i])):
            cost = cost[:(i+2)]
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
        cost[i+1] = np.sqrt(np.mean(np.power(X - X_hat, 2)))

        if plot_it:
            if i > 0:
                try:
                    h.close()
                except:
                    pass
            h = plot(W, H)
            h.suptitle(f'iteration {i}', fontsize=8)
            h.show()

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

def plot(W, H, cmap='gray_r', factor_cmap='Spectral'):
    N, K, L, T = get_shapes(W, H)

    data_recon = reconstruct(W, H)

    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[1, 4])
    ax_h = plt.subplot(gs[1])
    ax_w = plt.subplot(gs[2])
    ax_data = plt.subplot(gs[3])

    #plot W, H, and data_recon
    sns.heatmap(np.hstack(list(map(np.squeeze, np.split(W, K, axis=1)))), cmap=cmap, ax=ax_w, cbar=False)
    sns.heatmap(H, cmap=cmap, ax=ax_h, cbar=False)
    sns.heatmap(data_recon, cmap=cmap, ax=ax_data, cbar=False)

    #add dividing bars for factors of W and H
    factor_colors = sns.color_palette(factor_cmap, K)
    for k in np.arange(K):
        plt.sca(ax_w)
        start_w = k*L
        plt.plot([start_w, start_w], [0, N-1], '-', color=factor_colors[k])

        plt.sca(ax_h)
        plt.plot([0, T-1], [k, k], '-', color=factor_colors[k])

    return fig

