# Compact Model Parameter Extraction Framework

A Python framework for extracting semiconductor device compact model parameters from TCAD simulations.

## Current features:
* Fit diode model (saturation current *I_s*, ideality factor *n* and series resistance *R_s*) to I-V data
* Perofrm global multi-temperature diode fitting using Arrhenius-style I_s(T) model with bandgap Eg
* Fit MOSFET Level 1 (Schichman-Hodges) model to extract threshold voltage *V_th*, transconductance *k_n*, and channel-length modulation *lambda*
* Generate synthetic I-V curves for diodes and MOSFETs
* Visualize fitted curves and current errors on linear or log scales
* Unit tests validating parameter extraction for both single-temperature and multi-temperature fits

## Setup

```bash
conda env create -f environment.yml
conda activate compact-model-extraction
```

Then, run the Jupyter notebooks found in ```/examples/``` directory.

## Project structure
- 'src/models.py' - diode and MOSFET model implementation
- 'src/extraction.py' - parameter extraction logic
- 'src/visualization.py' - plotting helpers for fits and errors
- 'tests/' - unit tests
- 'examples/' - demonstration notebooks for model extraction

