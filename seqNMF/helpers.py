import numpy as np

def get_shapes(W, H):
    N = W.shape[0]
    K = W.shape[1]
    L = W.shape[2]
    T = H.shape[1]

    return N, K, L, T


def reconstruct(W, H):
    N, K, L, T = get_shapes(W, H)

    H = np.hstack((np.zeros([K, L]), H, np.zeros([K, L])))
    T += 2 * L
    X_hat = np.zeros([N, T])

    for t in np.arange(L):
        X_hat += np.dot(W[:, :, t], np.roll(H, t - 1, axis=1))

    return X_hat[:, L:-L]


def shift_factors(W, H):
    N, K, L, T = get_shapes(W, H)

    if L > 1:
        center = np.max([np.floor(L / 2), 1])
        Wpad = np.stack((np.zeros([N, K, L]), W, np.zeros([N, K, L])), axis=2)

        for i in np.arange(K):
            temp = np.sum(np.squeeze(W[:, i, :]), axis=0)
            # return temp, temp
            cmass = np.max(np.floor(np.sum(temp * np.arange(1, L + 1)) / np.sum(temp)), axis=0)
            Wpad[:, i, :] = np.roll(np.squeeze(Wpad[:, i, :]), center - cmass, axis=1)
            H[i, :] = np.roll(H[i, :], cmass - center, axis=0)

    return Wpad[:, :, L:-L], H


def compute_loadings_percent_power(V, W, H):
    K = H.shape[0]
    loadings = np.zeros([1, K])
    var_v = np.sum(np.pow(V, 2))

    for i in np.arange(K):
        WH = reconstruct(W[:, i, :], H[i, :])
        loadings[i] = np.divide(np.sum(np.multiply(2 * V.flatten(), WH.flatten()) - np.pow(WH.flatten(), 2)), var_v)

    loadings[loadings < 0] = 0
    return loadings

