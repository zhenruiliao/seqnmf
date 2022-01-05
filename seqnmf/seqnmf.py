from jax.scipy.signal import convolve2d as conv2
import jax, jax.numpy as jnp
import tqdm
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import gridspec
from .helpers import reconstruct, shift_factors, compute_loadings_percent_power, get_shapes, shifted_matrix_product


def update_W(W, H, X, Lambda, M, L, K, smooth_kernel, eps, lambda_OrthW, lambda_L1W):
    X_hat = reconstruct(W, H)
    X = jnp.where(M==0, X_hat, X)
    XHT = shifted_matrix_product(X,H.T,jnp.arange(L)-1,None,0)
    X_hat_HT = shifted_matrix_product(X_hat,H.T,jnp.arange(L)-1,None,0)
    XS = conv2(X, smooth_kernel, 'same')
    XS_HT = shifted_matrix_product(XS, H.T, jnp.arange(L)-1,None,0)
    dWWdW = jnp.dot(lambda_OrthW * jnp.sum(W, axis=2), 1. - jnp.eye(K))
    dRdW = Lambda * jax.vmap(lambda x: jnp.dot(x, 1-jnp.eye(K)))(XS_HT) + lambda_L1W + dWWdW
    return W * jnp.moveaxis(jnp.divide(XHT, X_hat_HT + dRdW + eps),0,2)


def seqnmf_iter(W, H, X, X_hat, Lambda, M, L, K, smooth_kernel, shift, eps, 
                W_fixed, lambda_OrthW, lambda_OrthH, lambda_L1W, lambda_L1H):
    
    WTX = shifted_matrix_product(W.T,X,-jnp.arange(L)+1,0,1).sum(0)
    WTX_hat = shifted_matrix_product(W.T,X_hat,-jnp.arange(L)+1,0,1).sum(0)          
    
    dRdH = jnp.dot(Lambda * (1 - jnp.eye(K)), conv2(WTX, smooth_kernel, 'same'))
    dHHdH = jnp.dot(lambda_OrthH * (1 - jnp.eye(K)), conv2(H, smooth_kernel, 'same'))
    dRdH += lambda_L1H + dHHdH

    H = H * jnp.divide(WTX, WTX_hat + dRdH + eps)
    
    W,H = jax.lax.cond(shift, shift_factors, lambda WH: WH, (W,H))
    W = W + eps*shift

    norms = jnp.sqrt(jnp.sum(jnp.power(H, 2), axis=1)).T
    H = jnp.dot(jnp.diag(jnp.divide(1., norms + eps)), H)
    W = jax.vmap(jnp.dot, in_axes=(2,None), out_axes=2)(W,jnp.diag(norms))
    
    update = lambda w: update_W(w, H, X, Lambda, M, L, K, smooth_kernel, eps, lambda_OrthW, lambda_L1W)
    W = jax.lax.cond(not W_fixed, update, lambda w: w, W)

    X_hat = reconstruct(W, H)
    X = jnp.where(M==0, X_hat, X)
    cost = jnp.sqrt(jnp.mean(jnp.power(X - X_hat, 2)))
    return W, H, X, X_hat, cost


def seqnmf(X, K=10, L=100, Lambda=.001, W_init=None, H_init=None,
           plot_it=False, max_iter=100, tol=-np.inf, shift=True, sort_factors=True,
           lambda_L1W=0, lambda_L1H=0, lambda_OrthH=0, lambda_OrthW=0, M=None, W_fixed=False):
    '''
    :param X: an N (features) by T (timepoints) data matrix to be factorized using seqNMF
    :param K: the (maximum) number of factors to search for; any unused factors will be set to all zeros
    :param L: the (maximum) number of timepoints to consider in each factor; any unused timepoints will be set to zeros
    :param Lambda: regularization parameter (default: 0.001)
    :param W_init: initial factors (if unspecified, use random initialization)
    :param H_init: initial per-timepoint factor loadings (if unspecified, initialize randomly)
    :param plot_it: if True, display progress in each update using a plot (default: False)
    :param max_iter: maximum number of iterations/updates
    :param tol: if cost is within tol of the average of the previous 5 updates, the algorithm will terminate (default: tol = -inf)
    :param shift: allow timepoint shifts in H
    :param sort_factors: sort factors by time
    :param lambda_L1W: regularization parameter for W (default: 0)
    :param lambda_L1H: regularization parameter for H (default: 0)
    :param lambda_OrthH: regularization parameter for H (default: 0)
    :param lambda_OrthW: regularization parameter for W (default: 0)
    :param M: binary mask of the same size as X, used to ignore a subset of the data during training (default: use all data)
    :param W_fixed: if true, fix factors (W), e.g. for cross validation (default: False)

    :return:
    :W: N (features) by K (factors) by L (per-factor timepoints) tensor of factors
    :H: K (factors) by T (timepoints) matrix of factor loadings (i.e. factor timecourses)
    :cost: a vector of length (number-of-iterations + 1) containing the initial cost and cost after each update (i.e. the reconstruction error)
    :loadings: the per-factor loadings-- i.e. the explanatory power of each individual factor
    :power: the total power (across all factors) explained by the full reconstruction
    '''
    assert np.all(X >= 0), 'all data values must be positive!'
    
    N = X.shape[0]
    T = X.shape[1] + 2 * L
    X = jnp.concatenate((jnp.zeros([N, L]), X, jnp.zeros([N, L])), axis=1)

    if W_init is None:
        W_init = jnp.array(np.max(X) * np.random.rand(N, K, L))
    if H_init is None:
        H_init = jnp.array(np.max(X) * np.random.rand(K, T) / np.sqrt(T / 3))
    if M is None:
        M = jnp.ones([N, T])

    W = W_init
    H = H_init

    X_hat = reconstruct(W, H)
    X = jnp.where(M==0, X_hat, X)

    smooth_kernel = jnp.ones([1, (2 * L) - 1])
    eps = jnp.max(X) * 1e-6
    last_time = False

    costs = np.zeros(max_iter + 1)
    costs[0] = jnp.sqrt(jnp.mean(jnp.power(X - X_hat, 2)))
    
    update = jax.jit(lambda W,H,X,X_hat,Lambda: seqnmf_iter(
        W, H, X, X_hat, Lambda, M, L, K, smooth_kernel, shift, eps,
        W_fixed, lambda_OrthW, lambda_OrthH, lambda_L1W, lambda_L1H))

    for i in tqdm.trange(max_iter):
        if (i == max_iter - 1) or ((i > 6) and (costs[i + 1] + tol) > np.mean(costs[i - 6:i])):
            costs = costs[:(i + 2)]
            last_time = True
            if i > 0: Lambda = 0
                
        W, H, X, X_hat, cost = update(W, H, X, X_hat, Lambda)
        costs[i] = cost
        

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

    X = X[:, L:-L]
    X_hat = X_hat[:, L:-L]
    H = H[:, L:-L]
    power = jnp.divide(jnp.sum(jnp.power(X, 2)) - jnp.sum(jnp.power(X - X_hat, 2)), jnp.sum(jnp.power(X, 2)))
    loadings = compute_loadings_percent_power(X, W, H)

    W = np.array(W)
    H = np.array(H)
    power = np.array(power)
    loadings = np.array(loadings)
    
    if sort_factors:
        inds = np.flip(np.argsort(loadings), 0)
        loadings = loadings[inds]
        W = W[:, inds, :]
        H = H[inds, :]
        
    return W, H, costs, loadings, power



def plot(W, H, cmap='gray_r', factor_cmap='Spectral'):
    '''
    :param W: N (features) by K (factors) by L (per-factor timepoints) tensor of factors
    :param H: K (factors) by T (timepoints) matrix of factor loadings (i.e. factor timecourses)
    :param cmap: colormap used to draw heatmaps for the factors, factor loadings, and data reconstruction
    :param factor_cmap: colormap used to distinguish individual factors
    :return f: matplotlib figure handle
    '''

    N, K, L, T = get_shapes(W, H)
    W, H = trim_shapes(W, H, N, K, L, T)

    data_recon = reconstruct(W, H)

    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[1, 4])
    ax_h = plt.subplot(gs[1])
    ax_w = plt.subplot(gs[2])
    ax_data = plt.subplot(gs[3])

    # plot W, H, and data_recon
    sns.heatmap(np.hstack(list(map(np.squeeze, np.split(W, K, axis=1)))), cmap=cmap, ax=ax_w, cbar=False)
    sns.heatmap(H, cmap=cmap, ax=ax_h, cbar=False)
    sns.heatmap(data_recon, cmap=cmap, ax=ax_data, cbar=False)

    # add dividing bars for factors of W and H
    factor_colors = sns.color_palette(factor_cmap, K)
    for k in np.arange(K):
        plt.sca(ax_w)
        start_w = k * L
        plt.plot([start_w, start_w], [0, N - 1], '-', color=factor_colors[k])

        plt.sca(ax_h)
        plt.plot([0, T - 1], [k, k], '-', color=factor_colors[k])

    return fig