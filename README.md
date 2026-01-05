# Compact Model Parameter Extraction Framework

A Python framework for extracting semiconductor device compact model parameters from TCAD simulations.

## Current features:
* Fit diode model to I-V data
* Perform global multi-temperature diode fitting using Arrhenius-style $I_s(T)$ model with bandgap $E_g$
* MOSFET Extraction
    * Fit Level 1 (Schichman-Hodges) model to single $I_d-V_{gs}$ curves
    * Extract $V_{th}$, $k_n$ and $\lambda$ simultaneously from a family of output characteristics $I_d-V_{ds}$
* Generate synthetic I-V curves for diodes and MOSFETs for testing
* Visualization
    * Plot diode fits (linear/log scales)
    * Plot MOSFET transfer characteristics $I_d-V_{gs}$
    * Plot MOSFET output characteristics families $I_d-V_{ds}$
* Unit tests for validating parameter extraction for both devices

## Setup

```bash
conda env create -f environment.yml
conda activate compact-model-extraction
```

Then, run the Jupyter notebooks found in ```/examples/``` directory.
* `examples/diode_extraction.ipynb`: Diode parameter extraction demo
* `examples/mosfet/extraction.ipynb`: MOSFET parameter extraction demo

## Project structure
- 'src/models.py' - diode and MOSFET model implementation
- 'src/extraction.py' - parameter extraction logic
- 'src/visualization.py' - plotting helpers for fits and errors
- 'tests/' - unit tests
- 'examples/' - demonstration notebooks for model extraction

