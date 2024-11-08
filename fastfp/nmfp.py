# -*- coding: utf-8 -*-
"""
Noise-marginalized Fp-statistic class and its function dependencies
"""
import scipy.constants as sc

# JAX imports
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

jax.config.update("jax_enable_x64", True)

# relative imports
from fastfp.utils import get_xCy

yr = sc.Julian_year
fyr = 1.0 / yr


@register_pytree_node_class
class NMFP(object):
    """
    Noise-marginalized Fp-statistic
    """

    def __init__(self, psrs, rn_sigs):
        self.psrs = psrs
        self.rn_sigs = rn_sigs

        self.toas = [psr.toas for psr in psrs]
        self.residuals = [psr.residuals for psr in psrs]

    def __call__(self, fgw, samples, Nvecs, Ts, TNTs):
        """Callable method
        """
        return self.calculate_nmfp(fgw, samples, Nvecs, Ts, TNTs)

    @jax.jit
    def _get_sigmas(self, pars, TNTs):
        sigmas = []
        for i, (rn_sig, TNT) in enumerate(zip(self.rn_sigs, TNTs)):
            phiinv = rn_sig.get_phiinv(pars)
            sigmas.append(TNT + jnp.diag(phiinv))
        return sigmas

    @jax.jit
    def calculate_nmfp(self, fgw, samples, Nvecs, Ts, TNTs):
        sigmas = self._get_sigmas(samples, TNTs)
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
        return (), (self.psrs, self.rn_sigs)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data, *children)


@register_pytree_node_class
class RN_container(object):
    """
    Container class for storing and updating per-pulsar
    red-noise covariance matrices
    """

    def __init__(self, psr, Ffreqs=None, ncomps=30, tm_marg=False):
        self.psr = psr
        self.ncomps = ncomps
        self.tm_marg = tm_marg

        self.rn_A_name = f"{psr.name}_red_noise_log10_A"
        self.rn_gam_name = f"{psr.name}_red_noise_gamma"

        if isinstance(Ffreqs, jax.Array):
            self.Ffreqs = Ffreqs
        # this will add the _create_freqarray to jit-compilation
        # which will slow things down
        else:
            self.Ffreqs = self._create_freqarray(psr, ncomps=ncomps)

        # need larger phi matrix if using non-marginalized timing model
        if not tm_marg:
            self.weights = jnp.ones(psr.Mmat.shape[1])
            self.phi_fn = self.get_phi_rn_tm
        else:
            self.phi_fn = self.get_phi_rn

    def _create_freqarray(self, psr, ncomps=30):
        Tspan = jnp.max(psr.toas) - jnp.min(psr.toas)
        f = 1.0 * jnp.arange(1, ncomps + 1) / Tspan
        Ffreqs = jnp.repeat(f, 2)
        return Ffreqs

    @jax.jit
    def _powerlaw(self, pars):
        df = jnp.diff(jnp.concatenate((jnp.array([0]), self.Ffreqs[::2])))
        return (
            self.Ffreqs ** (-pars[self.rn_gam_name])
            * (10 ** pars[self.rn_A_name]) ** 2
            / 12.0
            / jnp.pi**2
            * fyr ** (pars[self.rn_gam_name] - 3)
            * jnp.repeat(df, 2)
        )

    @jax.jit
    def get_phi_rn(self, pars):
        return self._powerlaw(pars)

    @jax.jit
    def get_phi_rn_tm(self, pars):
        rn_phi = self._powerlaw(pars)
        tm_phi = self.weights * 1e40
        return jnp.concatenate((tm_phi, rn_phi))

    @jax.jit
    def update_phi(self, pars):
        return self.phi_fn(pars)

    @jax.jit
    def get_phiinv(self, pars):
        phi = self.update_phi(pars)
        return 1.0 / phi

    def tree_flatten(self):
        return (self.Ffreqs,), (self.psr, self.ncomps, self.tm_marg)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        psr, ncomps, tm_marg = aux_data
        (Ffreqs,) = children
        return cls(psr, Ffreqs, ncomps, tm_marg)
