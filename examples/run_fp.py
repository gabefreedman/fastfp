# -*- coding: utf-8 -*-
"""
Basic script for running Fp-statistic analysis with fastfp.
"""

import pickle, json, time
import logging, argparse
import numpy as np

import jax

from fastfp.fastfp import Fp_jax
from fastfp.utils import initialize_pta, get_mats

def main(psrfile, noisefile, savefile):
    """
    savefile: name for json file containing output Fp-stat values
    """
    # setup logging
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    # if you're using a GPU (which you easily can with this code thanks to JAX!)
    # this will be a sanity check that you are actually utilizing it
    logger.info('Default XLA backend {}'.format(jax.default_backend()))

    # load pulsar data and noise dictionary
    with open(psrfile, 'rb') as f:
        psrs = pickle.load(f)
    
    with open(noisefile, 'rb') as f:
        noise = json.load(f)
    
    # if including a CURN or other CP, we should
    # set those parameters in the noise dictionary
    # here they are set to what I used in some simulations
    noise['gw_gamma'] = 13/3
    noise['gw_log10_A'] = np.log10(2e-15)

    # make PTA object
    pta = initialize_pta(psrs, noise, inc_cp=True, gwb_comps=30)

    # precompute important matrix products
    t_start = time.perf_counter()
    Nvecs, Ts, sigmainvs = get_mats(pta, noise)
    t_end = time.perf_counter()
    logger.info('Precompute matrix wall time: {0:.4f} s'.format(t_end - t_start))

    # initialize Fp-stat class
    Fp_obj = Fp_jax(psrs, pta)

    # create range of GW frequencies to parse over
    freqs = np.linspace(2e-9, 3e-7, 200)

    # run Fp-statistic calculation
    t_start = time.perf_counter()
    fn = jax.vmap(Fp_obj.calculate_Fp, in_axes=(0, None, None, None))
    fps = fn(freqs, Nvecs, Ts, sigmainvs)
    t_end = time.perf_counter()
    logger.info('Fp-statistic wall time: {0:.4f} s'.format(t_end - t_start))

    # create and save a dictionary of freq:fp pairs
    res = {freq: float(fp) for freq, fp in zip(freqs, fps)}
    with open('{}.json'.format(savefile), 'w') as f:
        json.dump(res, f)
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('psrfile', type=str, help='filename for pulsars pickle object')
    parser.add_argument('noisefile', type=str, help='filename for noise dictionary')
    parser.add_argument('savefile', type=str, help='filename for resulting Fp dictionary')
    
    kwargs = vars(parser.parse_args())
    main(**kwargs)