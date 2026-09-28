import jax
import jax.numpy as jnp
from dp_muon.optim import momentum_trajectory_workload_matrix
from exp12.adam import recurrence

def toeplitz_matrix(coef, T):
    lag = jnp.arange(T)[:,None]-jnp.arange(T)[None,:]
    return jnp.where((lag>=0)&(lag<len(coef)), coef[jnp.clip(lag,0,len(coef)-1)], 0.)

def filter_noise(coef, z):
    return sum(coef[k]*jnp.pad(z[:len(z)-k], ((k,0),)+((0,0),)*(z.ndim-1)) for k in range(len(coef)))

def momentum(c, T):
    return momentum_trajectory_workload_matrix(T, c.beta1, c.learning_rate, 0.)

def moment_matrix(T, beta, corrected=True):
    lag = jnp.arange(T)[:,None]-jnp.arange(T)[None,:]
    H = jnp.where(lag>=0, (1-beta)*beta**jnp.maximum(lag,0), 0.)
    return H/(1-beta**jnp.arange(1,T+1))[:,None] if corrected else H

def expected_v(coef, T, sigma, beta2):
    variance = sigma**2 * jnp.cumsum(jnp.pad(coef**2, (0,T-len(coef))))
    return moment_matrix(T,beta2) @ variance

def exact(coef, z, sigma, c, denominator=None):
    q = sigma*filter_noise(coef,z)
    u, _ = recurrence(q,c.beta1,c.beta2,c.adam_eps,denominator)
    return jnp.mean(jnp.sum(jnp.cumsum(u,axis=0)**2,axis=0))/len(q)

def mean_field(coef, T, sigma, c, denominator=None):
    S = toeplitz_matrix(coef,T)
    pre = 1/(jnp.sqrt(expected_v(coef,T,sigma,c.beta2))+c.adam_eps) if denominator is None else jnp.ones(T)/denominator
    U = pre[:,None] * (moment_matrix(T,c.beta1) @ (sigma*S))
    return jnp.sum(jnp.cumsum(U,axis=0)**2)/T

def momentum_objective(coef, T, c):
    return jnp.sum((momentum(c,T)@toeplitz_matrix(coef,T))**2)/T
