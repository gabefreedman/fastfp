# fastfp
Pulsar timing array Fp-statistic, now with JAX. It's pretty fast.

## Installation
Considering how minimal and speific-use-case this code is, I don't plan to formally host it on PyPi or conda for the forseeable future. For quick and painless installation, run the command below (Note: Since `enterprise`, and therefore `libstempo` is a dependency, this installation command may fail if you attempt to install to an environment that doesn't already have `libstempo` in it. Or maybe it'll be fine... who really knows these days).
```
pip install git+https://github.com/gabefreedman/fastfp.git
```
If you want to install with support for GPUs, install using the `gpu` option:
```
pip install "fastfp[gpu] @ git+https://github.com/gabefreedman/fastfp.git"
```

## Usage
See examples folder for both a Jupyter notebook and plain Python script detailing basic use.

## To-do
- ~~Noise marginalization~~
- ~~Additional installation instructions for GPU usage~~
- Separate get_xCy function for EcorrKernelNoise (block-diagonal)
- Include Fe-statistic?
