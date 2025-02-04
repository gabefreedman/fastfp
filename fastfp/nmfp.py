# -*- coding: utf-8 -*-
"""Noise-marginalized Fp-statistic class and its function dependencies,
written in JAX to allow for batching over CW frequencies as well as
multiple noise realization (or draws from an MCMC chain).

Can be run on GPUs for accelerated Fp-statistic calculations.
"""
# relative imports
import fastfp.constants as const
from fastfp.utils import get_xCy

# JAX imports
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

jax.config.update("jax_enable_x64", True)


@register_pytree_node_class
class NMFP(object):
    """Class for calculating the noise-marginalized Fp-statistic.
    This takes the calculation of the Fp-statistic in Ellis,
    Siemens, and Creighton (2012) and runs for multiple
    realizations of the pulsar red noise in the PTA. This is
    typically accomplished by taking random draws from an MCMC
    chain.

    The per-pulsar red-noise signals are kept in a set of separate
    classes written as JAX Pytrees. This allows the red-noise covariance
    update step of the noise-marginalization process to be JIT-compiled.

    :param psrs: A list of :class:`enterprise.pulsar.Pulsar` objects
        containing pulsar TOAs and residuals
    :type psrs: list
    :param rn_sigs: A list of :class:`fastfp.nmfp.RN_container` objects
        containing the per-pulsar red-noise signals
    :type rn_sigs: list
    """

    def __init__(self, psrs, rn_sigs):
        """Constructor method"""
        self.psrs = psrs
        self.rn_sigs = rn_sigs

        self.toas = [psr.toas for psr in psrs]
        self.residuals = [psr.residuals for psr in psrs]

    def __call__(self, fgw, samples, Nvecs, Ts, TNTs):
        """Callable method"""
        return self.calculate_nmfp(fgw, samples, Nvecs, Ts, TNTs)

    @jax.jit
    def _get_sigmas(self, pars, TNTs):
        """Calculate :math:`\\Sigma = B^{-1} + T^{T}N^{-1}T`, the
        portion of the total noise covariance that changes during
        updates of the red noise.

        :param pars: Dictionary of noise parameters
        :type pars: dict
        :param TNTs: List of :math:`T^{T}N^{-1}T` matrices for each pulsar
        :type TNTs: list
        :return: :math:`\\Sigma` matrix for each pulsar
        :type: list
        """
        sigmas = []
        for i, (rn_sig, TNT) in enumerate(zip(self.rn_sigs, TNTs)):
            phiinv = rn_sig.get_phiinv(pars)
            sigmas.append(TNT + jnp.diag(phiinv))
        return sigmas

    @jax.jit
    def calculate_nmfp(self, fgw, samples, Nvecs, Ts, TNTs):
        """Calculate Fp value for a given GW frequency and at a particular
        set of per-pulsar red noise parameters.

        :param fgw: Input GW frequency
        :type fgw: float
        :param samples: Dictionary of noise parameters
        :type samples: dict
        :param Nvecs: List of per-pulsar white-noise covariance matrices
        :type Nvecs: list
        :param Ts: List of per-pulsar basis matrices for Gaussian-process
            signals
        :type Ts: list
        :param TNTs: List of :math:`T^{T}N^{-1}T` matrices for each pulsar
        :type TNTs: list
        :return: :math:`F_{p}` value
        :rtype: float
        """
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
        """Method for flattening custom PyTree"""
        return (), (self.psrs, self.rn_sigs)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Method for reconstructing custom PyTree"""
        return cls(*aux_data, *children)


@register_pytree_node_class
class RN_container(object):
    """
    Container class for storing and updating per-pulsar
    red-noise covariance matrices. If CURN_container is provided,
    it will get added to all RN_container phi matrices at runtime.

    :param psr: A single pulsar data container
    :type psr: :class:`enterprise.pulsar.Pulsar`
    :param Ffreqs: Array of Fourier frequencies for red-noise
        basis model. (Note: if not provided, frequencies will
        be calculated using input pulsar's TOAs, and the calculation
        will be later be JIT-compiled along with any FP-statistic
        calculations, which will slow down runtime.)
    :type Ffreqs: array-like, optional.
    :param ncomps: Number of Fourier components for red-noise model
    :type ncomps: int, optional
    :param tm_marg: Flag for turning on the marginalizing of the
        timing model
    :type tm_marg: bool, optional
    """

    def __init__(
        self,
        psr,
        Ffreqs=None,
        ncomps=30,
        tm_marg=False,
        add_curn=False,
        curn_container=None,
    ):
        self.psr = psr
        self.ncomps = ncomps
        self.tm_marg = tm_marg
        self.add_curn = add_curn
        self.curn_container = curn_container

        self.rn_A_name = f"{psr.name}_red_noise_log10_A"
        self.rn_gam_name = f"{psr.name}_red_noise_gamma"

        if isinstance(Ffreqs, jax.Array):
            self.Ffreqs = Ffreqs
        else:
            self.Ffreqs = self._create_freqarray(psr, ncomps=ncomps)

        # need larger phi matrix if using non-marginalized timing model
        if not tm_marg:
            self.weights = jnp.ones(psr.Mmat.shape[1])
            if add_curn:
                self.phi_fn = self.get_phi_rn_tm_curn
            else:
                self.phi_fn = self.get_phi_rn_tm
        elif add_curn:
            self.phi_fn = self.get_phi_rn_curn
        else:
            self.phi_fn = self.get_phi_rn

    def _create_freqarray(self, psr, ncomps=30):
        """Create array of Fourier frequencies for red-noise model. Uses
        input pulsar to set Tspan.

        :param psr: A single pulsar data container
        :type psr: :class:`enterprise.pulsar.Pulsar`
        :param ncomps: Number of Fourier components for red-noise model
        :type ncomps: int, optional
        :return: Array of Fourier frequencies
        :rtype: array-like
        """
        Tspan = jnp.max(psr.toas) - jnp.min(psr.toas)
        f = 1.0 * jnp.arange(1, ncomps + 1) / Tspan
        Ffreqs = jnp.repeat(f, 2)
        return Ffreqs

    @jax.jit
    def _powerlaw(self, pars):
        """Power-law model for red-noise PSD.

        :param pars: Dictionary of noise parameters
        :type pars: dict
        :return: Power-law model evaluated for input parameters
        :rtype: array-like
        """
        df = jnp.diff(jnp.concatenate((jnp.array([0]), self.Ffreqs[::2])))
        return (
            self.Ffreqs ** (-pars[self.rn_gam_name])
            * (10 ** pars[self.rn_A_name]) ** 2
            / 12.0
            / jnp.pi**2
            * const.fyr ** (pars[self.rn_gam_name] - 3)
            * jnp.repeat(df, 2)
        )

    @jax.jit
    def get_phi_rn(self, pars):
        return self._powerlaw(pars)

    @jax.jit
    def get_phi_rn_curn(self, pars):
        rn_phi = self._powerlaw(pars)
        curn_phi = self.curn_container.get_phi_curn(pars)
        return rn_phi.at[: curn_phi.shape[0]].add(curn_phi)

    @jax.jit
    def get_phi_rn_tm(self, pars):
        rn_phi = self._powerlaw(pars)
        tm_phi = (
            self.weights * 1e-14 * len(self.psr.toas)
        )  # variance 1e-14 from utils.py
        return jnp.concatenate((tm_phi, rn_phi))

    @jax.jit
    def get_phi_rn_tm_curn(self, pars):
        rn_phi = self._powerlaw(pars)
        tm_phi = (
            self.weights * 1e-14 * len(self.psr.toas)
        )  # variance 1e-14 from utils.py
        curn_phi = self.curn_container.get_phi_curn(pars)
        return jnp.concatenate((tm_phi, rn_phi.at[: curn_phi.shape[0]].add(curn_phi)))

    @jax.jit
    def update_phi(self, pars):
        """Calculate red-noise prior covariance matrix.

        :param pars: Dictionary of noise parameters
        :type pars: dict
        :return: Red-noise prior covariance matrix
        :rtype: array-like
        """
        return self.phi_fn(pars)

    @jax.jit
    def get_phiinv(self, pars):
        """Calculate inverse of red-noise prior covariance matrix.

        :param pars: Dictionary of noise parameters
        :type pars: dict
        :return: Inverse of red-noise prior covariance matrix
        :rtype: array-like
        """
        phi = self.update_phi(pars)
        return 1.0 / phi

    def tree_flatten(self):
        """Method for flattening custom PyTree"""
        return (self.Ffreqs,), (
            self.psr,
            self.ncomps,
            self.tm_marg,
            self.add_curn,
            self.curn_container,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Method for reconstructing custom PyTree"""
        psr, ncomps, tm_marg, add_curn, curn_container = aux_data
        (Ffreqs,) = children
        return cls(psr, Ffreqs, ncomps, tm_marg, add_curn, curn_container)


@register_pytree_node_class
class CURN_container(object):
    """
    Container class for storing and updating common
    uncorrelated red-noise covariance matrices

    :param psr: A single pulsar data container
    :type psr: :class:`enterprise.pulsar.Pulsar`
    :param Ffreqs: Array of Fourier frequencies for red-noise
        basis model. (Note: if not provided, frequencies will
        be calculated using input pulsar's TOAs, and the calculation
        will be later be JIT-compiled along with any FP-statistic
        calculations, which will slow down runtime.)
    :type Ffreqs: array-like, optional.
    :param ncomps: Number of Fourier components for red-noise model
    :type ncomps: int, optional
    :param tm_marg: Flag for turning on the marginalizing of the
        timing model
    :type tm_marg: bool, optional
    """

    def __init__(self, Ffreqs):

        self.rn_A_name = "gw_log10_A"
        self.rn_gam_name = "gw_gamma"

        self.Ffreqs = Ffreqs

        self.phi_fn = self.get_phi_curn

    @jax.jit
    def _powerlaw(self, pars):
        """Power-law model for red-noise PSD.

        :param pars: Dictionary of noise parameters
        :type pars: dict
        :return: Power-law model evaluated for input parameters
        :rtype: array-like
        """
        df = jnp.diff(jnp.concatenate((jnp.array([0]), self.Ffreqs[::2])))
        return (
            self.Ffreqs ** (-pars[self.rn_gam_name])
            * (10 ** pars[self.rn_A_name]) ** 2
            / 12.0
            / jnp.pi**2
            * const.fyr ** (pars[self.rn_gam_name] - 3)
            * jnp.repeat(df, 2)
        )

    @jax.jit
    def get_phi_curn(self, pars):
        return self._powerlaw(pars)

    @jax.jit
    def update_phi(self, pars):
        """Calculate red-noise prior covariance matrix.

        :param pars: Dictionary of noise parameters
        :type pars: dict
        :return: Red-noise prior covariance matrix
        :rtype: array-like
        """
        return self.phi_fn(pars)

    @jax.jit
    def get_phiinv(self, pars):
        """Calculate inverse of red-noise prior covariance matrix.

        :param pars: Dictionary of noise parameters
        :type pars: dict
        :return: Inverse of red-noise prior covariance matrix
        :rtype: array-like
        """
        phi = self.update_phi(pars)
        return 1.0 / phi

    def tree_flatten(self):
        """Method for flattening custom PyTree"""
        return (self.Ffreqs,), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Method for reconstructing custom PyTree"""
        return cls(*aux_data, *children)
