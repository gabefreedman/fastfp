# -*- coding: utf-8 -*-
"""Module holding all the utility functions and other
modifications to set up PTA objects and run an
Fp-statistic analysis
"""
# enterprise imports
from enterprise.signals.parameter import Constant, function
from enterprise.signals.white_signals import MeasurementNoise
from enterprise.signals.gp_signals import BasisGP, get_timing_model_basis
from enterprise.signals.signal_base import PTA

from enterprise_extensions.model_utils import get_tspan
from enterprise_extensions.blocks import (
    red_noise_block,
    common_red_noise_block,
    white_noise_block,
)

# jax imports
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@jax.jit
def get_xCy(Nvec, T, sigma, x, y):
    """Compute :math:`x^{T}C^{-1}y`, where
    :math:`C = N + TBT^{T}`. This function does
    not apply for the case where :math:`N` is block-diagonal
    (i.e., ECORR modeled as white-noise)

    :param Nvec: White-noise covariance matrix for a single pulsar
    :type Nvec: array-like
    :param T: Concatenated basis matrices for Gaussian-process signals
        for a single pulsar
    :type T: array-like
    :param sigma: :math:`\\Sigma = B^{-1} + T^{T}N^{-1}T`
        for a single pulsar, with :math:`B` denoting the red-noise
        covariance matrix
    :type sigma: array-like
    :param x: Input vector
    :type x: array-like
    :param y: Another input vector
    :type y: array-like
    :return: :math:`x^{T}C^{-1}y`
    :rtype: float
    """
    Nx = jnp.array(x / Nvec)
    Ny = jnp.array(y / Nvec)
    TNx = jnp.dot(T.T, Nx)
    TNy = jnp.dot(T.T, Ny)
    xNy = jnp.dot(x.T, Ny)
    return xNy - TNx @ jnp.linalg.solve(sigma, TNy)


def get_mats_fp(pta, noise):
    """Precompute a bunch of matrix products or vectors
    that are needed for the base Fp-statistic calculation

    :param pta: An :class:`enterprise.signal_base.PTA` object
    :type pta: :class:`enterprise.signal_base.PTA`
    :param noise: Dictionary containing values for all noise parameters
        in the input PTA object
    :type noise: dict
    :return: tuple (Nvecs, Ts, sigmas), containing the necessary
        noise covariance and basis matrices for calculating an
        :math:`F_{p}`-statistic
    :rtype: tuple
    """

    phiinvs = pta.get_phiinv(noise)
    TNTs = pta.get_TNT(noise)
    Nvecs = pta.get_ndiag(noise)
    Ts = pta.get_basis(noise)
    sigmas = [(TNT + jnp.diag(phiinv)) for TNT, phiinv in zip(TNTs, phiinvs)]

    return Nvecs, Ts, sigmas


def get_mats_nmfp(pta, noise):
    """Precompute a bunch of matrix products or vectors
    that are needed for the noise-marginalized Fp-statistic
    calculation

    :param pta: An :class:`enterprise.signal_base.PTA` object
    :type pta: :class:`enterprise.signal_base.PTA`
    :param noise: Dictionary containing values for all noise parameters
        in the input PTA object
    :type noise: dict
    :return: tuple (TNTs, Nvecs, Ts), containing the necessary
        noise covariance matrices and matrix products for calculating
        :math:`F_{p}`-statistic
    :rtype: tuple
    """

    TNTs = pta.get_TNT(noise)
    Nvecs = pta.get_ndiag(noise)
    Ts = pta.get_basis(noise)

    return TNTs, Nvecs, Ts


# A small change to the TimingModel class suggested
# by Aaron and Shashwat that adds a variance kwarg
@function
def tm_prior(weights, toas, variance=1e-14):
    return weights * variance * len(toas)


def TimingModel(
    coefficients=False,
    name="linear_timing_model",
    use_svd=False,
    normed=True,
    prior_variance=1e-14,
):
    """Class factory for linear timing model signals."""

    basis = get_timing_model_basis(use_svd, normed)
    prior = tm_prior(variance=prior_variance)

    BaseClass = BasisGP(prior, basis, coefficients=coefficients, name=name)

    class TimingModel(BaseClass):
        signal_type = "basis"
        signal_name = "linear timing model"
        signal_id = name + "_svd" if use_svd else name

    return TimingModel


def initialize_pta(psrs, noise, inc_cp=True, rn_comps=30, gwb_comps=30, simple_wn=True):
    """
    simple_wn: Use an incredibly basic (EFAC=1.0) representation
    of white-noise, usually associated with simulated PTA datasets

    :param psrs: A list of :class:`enterprise.pulsar.Pulsar` objects
    :type psrs: list
    :param noise: Dictionary containing values for all noise parameters
        in the input PTA object
    :type noise: dict
    :param inc_cp: Include a common-process red-noise signal, currently
        implemented as uncorrelated, defaults to `True`
    :type inc_cp: bool, optional
    :param rn_comps: Number of frequencies for per-pulsar red-noise
        model, defaults to 30
    :type rn_comps: int, optional
    :param gwb_comps: Number of frequencies for common-process
        red-noise model, defaults to 30
    :type gwb_comps: int, optional
    :param simple_wn: Flag for setting per-pulsar white-noise
        model to an incredibly simple EFAC=1.0 signal, defaults to
        `True`. This is occasionally a choice when making a large
        suite of simulated PTA datasets for exploratory analyses
    :type simple_wn: bool, optional
    :return: An :class:`enterprise.signal_base.PTA` object with
        full input signal model and noise dictionary
    :rtype: :class:`enterprise.signal_base.PTA`
    """
    Tspan = get_tspan(psrs)
    tm = TimingModel(use_svd=True)
    if simple_wn:
        efac = Constant(1.0)
        wn = MeasurementNoise(efac=efac)
    else:
        wn = white_noise_block(inc_ecorr=True, gp_ecorr=True)
    rn = red_noise_block(Tspan=Tspan, components=rn_comps)
    # full signal
    s = tm + wn + rn
    if inc_cp:
        s += common_red_noise_block(Tspan=Tspan, components=gwb_comps)
    # initialize PTA
    model = [s(psr) for psr in psrs]
    pta = PTA(model)
    pta.set_default_params(noise)
    return pta
