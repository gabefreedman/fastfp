# -*- coding: utf-8 -*-
"""
Fp-statistic calculation and its function dependencies
but now all written in JAX. Allows for batching with jax.vmap,
effectively turning the main loop over GW frequencies into
a single matrix operation.

Also since this utilizes JAX, it should (in theory) be easily
portable to GPUs.
"""
# JAX imports
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
jax.config.update('jax_enable_x64', True)

# relative imports
from fastfp.utils import get_xCy

@register_pytree_node_class
class Fp_jax(object):
    """
    Try to make a PyTree container for Fp-statistic calculation
    """
    def __init__(self, psrs, pta):
        self.psrs = psrs
        self.pta = pta

        self.toas = [psr.toas for psr in psrs]
        self.residuals = [psr.residuals for psr in psrs]
    
    @jax.jit
    def calculate_Fp(self, fgw, Nvecs, Ts, sigmainvs):
        N = jnp.zeros(2)
        M = jnp.zeros((2,2))
        fstat = 0
        for (Nvec, T, sigmainv, toa, resid) in zip(Nvecs, Ts, sigmainvs, self.toas, self.residuals):
            ntoa = toa.shape[0]

            A = jnp.zeros((2, ntoa))
            A = A.at[0, :].set(1 / fgw ** (1 / 3) * jnp.sin(2 * jnp.pi * fgw * toa))
            A = A.at[1, :].set(1 / fgw ** (1 / 3) * jnp.cos(2 * jnp.pi * fgw * toa))

            ip1 = get_xCy(Nvec, T, sigmainv, A[0, :], resid)
            ip2 = get_xCy(Nvec, T, sigmainv, A[1, :], resid)
            N = jnp.array([ip1, ip2])

            M = M.at[0,0].set(get_xCy(Nvec, T, sigmainv, A[0,:], A[0,:]))
            M = M.at[0,1].set(get_xCy(Nvec, T, sigmainv, A[0,:], A[1,:]))
            M = M.at[1,0].set(get_xCy(Nvec, T, sigmainv, A[1,:], A[0,:]))
            M = M.at[1,1].set(get_xCy(Nvec, T, sigmainv, A[1,:], A[1,:]))

            Minv = jnp.linalg.pinv(M)
            fstat += 0.5 * jnp.dot(N, jnp.dot(Minv, N))

        return fstat

    def tree_flatten(self):
        return (), (self.psrs, self.pta)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data, *children)