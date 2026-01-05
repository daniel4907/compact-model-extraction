# Compact Model Parameter Extraction Framework

A Python framework for extracting semiconductor device compact model parameters from TCAD simulations.

## Current features:
* Fit diode model (saturation current *I_s*, ideality factor *n* and series resistance *R_s*) to I-V data
* Generate synthetic diode I-V curves for testing single-temperatuire and multi-temperature cases
* Perform global multi-temperature fitting using an Arrhenius-style I_s(T) model with bandgap Eg
* Visualize fitted curves and current errors on linear or log scales
* Unit tests validating parameter extraction for both single-temperature and multi-temperature fits

## Setup

```bash
conda env create -f environment.yml
conda activate compact-model-extraction
```

Then, run the Jupyter notebook found in ```/examples/diode_extraction.ipynb```

## Project structure
- 'src/models.py' - diode model implementation
- 'src/extraction.py' - parameter extraction
- 'src/visualization.py' - plotting helpers for fits and errors
- 'tests/' - unit tests
- 'examples/diode_extraction.ipynb' - single and multi-temperature diode I-V extraction demo

