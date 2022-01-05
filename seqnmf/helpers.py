import jax, jax.numpy as jnp
import numpy as np



def get_shapes(W, H):
    N = W.shape[0]
    T = H.shape[1]
    K = W.shape[1]
    L = W.shape[2]
    return N, K, L, T

def trim_shapes(W, H, N, K, L, T):
    return W[:N, :K, :L], H[:K, :T]

def reconstruct(W, H):
    N, K, L, T = get_shapes(W, H)
    H = jnp.hstack((jnp.zeros([K, L]), H, jnp.zeros([K, L])))
    X_hat = jax.vmap(lambda t: jnp.dot(W[:, :, t], jnp.roll(H, t - 1, axis=1)))(jnp.arange(L)).sum(0)
    return X_hat[:, L:-L]


def shift_factors(WH):
    W,H = WH
    N, K, L, T = get_shapes(W, H)
    center = jnp.maximum(jnp.floor(L / 2), 1)
    Wpad = jnp.concatenate((jnp.zeros([N, K, L]), W, jnp.zeros([N, K, L])), axis=2)
    
    def shift_row(wpad,w,h):
        temp = jnp.sum(jnp.squeeze(w), axis=0)
        cmass = jnp.floor(jnp.sum(temp * jnp.arange(1, L + 1)) / jnp.sum(temp))
        shift = (center - cmass).astype(int)
        wpad = jnp.roll(jnp.squeeze(wpad), shift, axis=1)
        h = jnp.roll(h, shift, axis=0)
        return wpad,h
    
    Wpad,H = jax.vmap(shift_row, in_axes=(1,1,0), out_axes=(1,0))(Wpad,W,H)
    return Wpad[:, :, L:-L], H


def compute_loadings_percent_power(V, W, H):
    N, K, L, T = get_shapes(W, H)
    W, H = trim_shapes(W, H, N, K, L, T)
    var_v = jnp.sum(np.power(V, 2))
    WH = jax.vmap(reconstruct, in_axes=(1,0))(W[:,:,None,:],H[:,None,:])
    loadings = (2*V.flatten()[None]*WH.reshape(K,-1)-WH.reshape(K,-1)**2).sum(1) / var_v
    return loadings * (loadings > 0)
    
def shifted_matrix_product(A,B,shifts,in_axis,shift_axis):
    return jax.vmap(lambda a,t: jnp.dot(a,jnp.roll(B,t,axis=shift_axis)), in_axes=(in_axis,0))(A,shifts)