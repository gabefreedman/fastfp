# fastfp
Pulsar timing array Fp-statistic, now with JAX. It's pretty fast.

## Installation
Installation can be done simply via pip.
```
pip install fastfp
```
If you want to install with support for GPUs, use the `gpu` extra option.
```
pip install "fastfp[gpu]"
```
If you want to install for development work, use the `dev` extra option.
```
pip install "fastfp[dev]"
```

## Usage
See examples folder for a Jupyter notebook and plain Python scripts detailing basic use of both the single and noise-marginalized Fp-statistic.

## To-do
- Separate get_xCy function for EcorrKernelNoise (block-diagonal)
- Include Fe-statistic?
