# -*- coding: utf-8 -*-
"""Fp-statistic calculation and its function dependencies
but now all written in JAX. Allows for batching with jax.vmap,
effectively turning the main loop over GW frequencies into
a single matrix operation.

Also since this utilizes JAX, it should (in theory) be easily
portable to GPUs.
"""
# relative imports
from fastfp.utils import get_xCy

# JAX imports
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

jax.config.update("jax_enable_x64", True)


@register_pytree_node_class
class FastFp(object):
    """Class for calculating the Fp-statistic, an incoherent
    detection statistic for continuous gravitational searches
    in pulsar timing array data. Follows the derivation in
    Ellis, Siemens, and Creighton (2012). This is essentially a
    rewrite of the :class:`enterprise_extensions.frequentist.FpStat`
    class using JAX for the matrix operations.

    :param psrs: A list of :class:`enterprise.pulsar.Pulsar` objects
        containing pulsar TOAs and residuals
    :type psrs: list
    :param pta: An :class:`enterprise.signal_base.PTA` object loaded
        with user-defined white- and red-noise signals, and
        optionally any common-process red-noise signals
    :type pta: :class:`enterprise.signal_base.PTA`
    """

    def __init__(self, psrs, pta):
        """Constructor method"""
        self.psrs = psrs
        self.pta = pta

        self.toas = [psr.toas for psr in psrs]
        self.residuals = [psr.residuals for psr in psrs]

    def __call__(self, fgw, Nvecs, Ts, sigmas):
        """Callable method"""
        return self.calculate_Fp(fgw, Nvecs, Ts, sigmas)

    @jax.jit
    def calculate_Fp(self, fgw, Nvecs, Ts, sigmas):
        """Calculate Fp value for a given GW frequency.

        :param fgw: Input GW frequency
        :type fgw: float
        :param Nvecs: List of per-pulsar white-noise covariance matrices
        :type Nvecs: list
        :param Ts: List of per-pulsar basis matrices for Gaussian-process
            signals
        :type Ts: list
        :param sigmas: List of :math:`\\Sigma` defined as
            :math:`\\Sigma = B^{-1} + T^{T}N^{-1}T`, with
            :math:`B` denoting the red-noise covariance matrix
        :type sigmas: list
        :return: :math:`F_{p}` value
        :rtype: float
        """
        N = jnp.zeros(2)
        M = jnp.zeros((2, 2))
        fstat = 0
        for Nvec, T, sigma, toa, resid in zip(
            Nvecs, Ts, sigmas, self.toas, self.residuals
        ):
            ntoa = toa.shape[0]

            A = jnp.zeros((2, ntoa))
            A = A.at[0, :].set(1 / fgw ** (1 / 3) * jnp.sin(2 * jnp.pi * fgw * toa))
            A = A.at[1, :].set(1 / fgw ** (1 / 3) * jnp.cos(2 * jnp.pi * fgw * toa))

            ip1 = get_xCy(Nvec, T, sigma, A[0, :], resid)
            ip2 = get_xCy(Nvec, T, sigma, A[1, :], resid)
            N = jnp.array([ip1, ip2])

            M = M.at[0, 0].set(get_xCy(Nvec, T, sigma, A[0, :], A[0, :]))
            M = M.at[0, 1].set(get_xCy(Nvec, T, sigma, A[0, :], A[1, :]))
            M = M.at[1, 0].set(get_xCy(Nvec, T, sigma, A[1, :], A[0, :]))
            M = M.at[1, 1].set(get_xCy(Nvec, T, sigma, A[1, :], A[1, :]))

            fstat += 0.5 * jnp.dot(N, jnp.linalg.solve(M, N))

        return fstat

    def tree_flatten(self):
        """Method for flattening custom PyTree"""
        return (), (self.psrs, self.pta)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Method for reconstructing custom PyTree"""
        return cls(*aux_data, *children)
