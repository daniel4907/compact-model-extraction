# Compact Model Parameter Extraction Framework

A Python framework for extracting semiconductor device compact model parameters from TCAD simulations.

## Current features:

### Diode Characteristics
* Extract saturation current $I_s$, ideality factor $n$, and series resistance $R_s$ from individual diode I-V curves
* **Multi-Temperature Analysis**
    * Perform global parameter extraction across a range of temperatures
    * Extract semiconductor bandgap energy $E_g$ using physics-based scaling laws
    * Visualize saturation current as a function of temperature

### MOSFET Characteristics
* Level 1 model (Schichman-Hodges): supports cutoff, triode, and saturation regions
* Extract threshold voltage and transconductance from transfer characteristics
* Extract channel length modulation by fitting family of output characteristic curves simultaneously
* Support for semi-log plotting to verify subthreshold behavior

### Utilities
* Create noisy, realistic I-V datasets for testing algorithms
* Automated plotting for fits, residuals, and parameter trends
* Uses non-linear least squares (Trust Region Reflective algorithm) with physical bounds constraints

## Setup
1. Clone the repository to your local machine
2. Create the conda environment:
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

