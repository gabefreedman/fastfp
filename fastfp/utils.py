# -*- coding: utf-8 -*-
"""
Module holding all the utility functions and other
modifications to set up PTA objects and run an
Fp-statistic analysis
"""
# enterprise imports
from enterprise.signals.parameter import Constant, function
from enterprise.signals.white_signals import MeasurementNoise
from enterprise.signals.gp_signals import BasisGP, get_timing_model_basis
from enterprise.signals.signal_base import PTA

from enterprise_extensions.model_utils import get_tspan
from enterprise_extensions.blocks import red_noise_block, common_red_noise_block, white_noise_block

# jax imports
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


@jax.jit
def get_xCy(Nvec, T, sigmainv, x, y):
    # Get x^T C^{-1} y
    # This won't work if we're using kernel ECORR
    # (i.e., block-diagonal instead of diagonal white-noise matrix)
    Nx = jnp.array(x / Nvec)
    Ny = jnp.array(y / Nvec)
    TNx = jnp.dot(T.T, Nx)
    TNy = jnp.dot(T.T, Ny)
    xNy = jnp.dot(x.T, Ny)
    return xNy - TNx @ sigmainv @ TNy

def get_mats(pta, noise):
    """
    Precompute a bunch of matrix products or vectors
    that will be fed into the Fp-statistic object
    """

    phiinvs = pta.get_phiinv(noise)
    TNTs = pta.get_TNT(noise)
    Nvecs = pta.get_ndiag(noise)
    Ts = pta.get_basis(noise)
    sigmainvs = [jnp.linalg.pinv(TNT + jnp.diag(phiinv)) for TNT, phiinv in zip(TNTs, phiinvs)]

    return Nvecs, Ts, sigmainvs


# A small change to the TimingModel class suggested
# by Aaron and Shashwat that adds a variance kwarg
@function
def tm_prior(weights, toas, variance=1e-14):
    return weights * variance * len(toas)

def TimingModel(coefficients=False, name="linear_timing_model",
                use_svd=False, normed=True, prior_variance=1e-14):
    """Class factory for linear timing model signals."""

    basis = get_timing_model_basis(use_svd, normed)
    prior = tm_prior(variance=prior_variance)

    BaseClass = BasisGP(prior, basis, coefficients=coefficients, name=name)

    class TimingModel(BaseClass):
        signal_type = "basis"
        signal_name = "linear timing model"
        signal_id = name + "_svd" if use_svd else name

    return TimingModel


# Main function for creating PTA object
# with desired signal properties
def initialize_pta(psrs, noise, inc_cp=True, rn_comps=30, gwb_comps=30,
                   simple_wn=True):
    """
    simple_wn: Use an incredibly basic (EFAC=1.0) representation
    of white-noise, usually associated with simulated PTA datasets
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