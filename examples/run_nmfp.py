# -*- coding: utf-8 -*-
"""
Script for running a noise-marginalized Fp-statistic
analysis with fastfp. The input samples should come
from a prior MCMC run.
"""

import pickle
import json
import time
import logging
import argparse
import numpy as np

# enterprise imports
from enterprise_extensions.model_utils import get_tspan

# JAX imports
import jax
import jax.numpy as jnp

# fastfp imports
from fastfp.nmfp import NMFP, RN_container
from fastfp.utils import initialize_pta, get_mats_nmfp


def create_freqarray(Tspan, ncomps=30):
    """Create the array of Fourier basis frequencies used for
    setting the red-noise parameters. Can be calculated based
    on the timespan of the entire PTA or inidividually for each
    pulsar.
    """
    f = 1.0 * jnp.arange(1, ncomps + 1) / Tspan
    Ffreqs = jnp.repeat(f, 2)
    return Ffreqs


def setup_fp_model(psrs, Tspan=None, nrncomps=30):
    """Create the NMFP object and its component red-noise containers.
    If Tspan is None, the red-noise bases are set inidividually. If
    Tspan is a set value, all pulsar red noises bases are set to be
    the same.
    """
    if not Tspan:
        Ffreqs = []
        for psr in psrs:
            Tspan = np.max(psr.toas) - np.min(psr.toas)
            Ffreqs.append(create_freqarray(Tspan, ncomps=nrncomps))
        rn_objs = [RN_container(psr, Ffreqs=Ffreqs[i]) for i, psr in enumerate(psrs)]
    else:
        Ffreqs = create_freqarray(Tspan, ncomps=nrncomps)
        rn_objs = [RN_container(psr, Ffreqs=Ffreqs) for psr in psrs]

    nmfp = NMFP(psrs, rn_objs)
    return nmfp


def map_params(pta, xs):
    """A rewrite of the pta.map_params function that
    can map 2D arrays of parameter samples.
    """
    if xs.ndim > 1:
        ret = {}
        ct = 0
        for p in pta.params:
            ret[p.name] = xs[ct, :]
            ct += 1
    else:
        ret = pta.map_params(xs)
    return ret


def main(
    psrfile,
    noisefile,
    chainfile,
    savefile,
    ncwfreqs=100,
    nsamples=1000,
    batch_size=100,
):
    # set up logging
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    # check if JAX is using a GPU (perhaps you want it to)
    logger.info("Default XLA backend {}".format(jax.default_backend()))

    # log initial conditions
    logger.info(f"number of CW frequencies: {ncwfreqs}")
    logger.info(f"number of samples: {nsamples}")
    logger.info(f"batch_size: {batch_size}")

    # load pulsar objects, noise dictionary, and MCMC chain
    with open(psrfile, "rb") as f:
        psrs = pickle.load(f)

    with open(noisefile, "r") as f:
        noise = json.load(f)

    chain = np.loadtxt(chainfile)
    burn = int(0.25 * chain.shape[0])

    # if including a CURN or other CP, we should
    # set those parameters in the noise dictionary
    # here they are set to what I used in some simulations
    noise["gw_gamma"] = 13 / 3
    noise["gw_log10_A"] = np.log10(2e-15)

    # create PTA and NMFP object
    Tspan = get_tspan(psrs)
    pta = initialize_pta(psrs, noise, inc_cp=False)
    nmfp = setup_fp_model(psrs, Tspan=Tspan)

    # precompute some matrices
    t_start = time.perf_counter()
    TNTs, Nvecs, Ts = get_mats_nmfp(pta, noise)
    logger.info(f"Precompute matrix wall time: {time.perf_counter() - t_start:.2f} s")

    # set array of CW frequencies
    freqs = jnp.linspace(2e-9, 3e-7, ncwfreqs)

    # generate array of RN samples
    nbatches = int(nsamples / batch_size)
    rns_full = jnp.zeros((len(pta.params), nsamples))
    idxs = np.random.choice(range(burn, chain.shape[0]), nsamples, replace=False)
    for j, idx in enumerate(idxs):
        rns_full = rns_full.at[:, j].set(chain[idx, :-4])

    # batch the RN samples
    samples_batched = []
    for i in range(nbatches):
        start = i * batch_size
        end = start + batch_size
        samples_batched.append(map_params(pta, rns_full[:, idxs[start:end]]))

    # double vmap: one for frequencies and one for red noise realizations
    t_start = time.perf_counter()
    vmap_f = jax.vmap(nmfp, in_axes=(0, None, None, None, None))
    vmap_g = jax.vmap(vmap_f, in_axes=(None, 0, None, None, None))
    nmfp_batched = []
    for i in range(nbatches):
        nmfp_batched.append(vmap_g(freqs, samples_batched[i], Nvecs, Ts, TNTs))
    nmfp_vals = np.vstack(nmfp_batched)
    logger.info(
        f"Noise marginalized Fp-statistic wall time: {time.perf_counter() - t_start:.2f} s"
    )

    with open(f"res/{savefile}.npy", "wb") as f:
        np.save(f, nmfp_vals)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("psrfile", type=str, help="filepath for pulsars pickle object")
    parser.add_argument("noisefile", type=str, help="filepath for noise dictionary")
    parser.add_argument("chainfile", type=str, help="filepath for MCMC chain")
    parser.add_argument(
        "savefile", type=str, help="filepath for output Fp values array"
    )
    parser.add_argument(
        "--ncwfreqs",
        type=int,
        default=100,
        required=False,
        help="number of CW frequencies to calculate at",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=1000,
        required=False,
        help="number of red noise draws from the MCMC chain",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        required=False,
        help="batch size for JAX vmapping",
    )

    kwargs = vars(parser.parse_args())
    main(**kwargs)
