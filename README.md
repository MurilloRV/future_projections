# **`future projections`** repository

This repository contains the code used for the Higgstools and BSM future collider studies

## Repository Structure:

### `model_scan_notebooks/`

Here are the jupyter notebooks that actually perform the studies. 
* `IDM_scan.ipynb` $\rightarrow$ Main notebook for the IDM studies. Reads the several IDM data files (all including the momentum dependence of the HZZ coupling) and obtains the benchmark points for the HEPfit studies. Also generates the $\kappa_{Zh}^{240}$ vs. $\kappa_{Zh}^{365}$ plots, including the IDM BPs as well as the predictions from HEPfit. Also includes, at the beginning, the functionality of `Z2_SSM_scan.ipynb`, but for the IDM instead of the Z2SSM

* `IDM_scan_plot.ipynb` $\rightarrow$ Shorter, cleaner version of `IDM_scan.ipynb`, containing only the basic code to generate the $\kappa_{Zh}^{240}$ vs. $\kappa_{Zh}^{365}$ plots for the IDM BPs

* `IDMscans_clean_Johannes.ipynb` $\rightarrow$ Johannes' python notebook to obtain the IDM data files

* `Z2_SSM_scan.ipynb` $\rightarrow$ Original notebook for Z2SSM studies using Higgstools. All Higgs couplings here are constant (i.e. evaluated at zero external momenta). Produces $\kappa_{\lambda}$ vs. $\kappa_{Z}$ plots to illustrate the future collider projections, which in turn are implemented in the `future_projections` branch of `hsdataset`. Reads a set of Z2SSM parameter points and determines the likelihood maximum for this set. Includes also tests to see the effect of modifying the projected precision of $\kappa_{\lambda}$ at the HL-LHC, as well as modifying the central values of the `higgssignals` likelihood according to the predictions of certain Z2SSM benchmark points

* `Z2_SSM_scan_energy_dependence.ipynb` $\rightarrow$ Analogous to `IDM_scan_plot.ipynb`, but for Z2SSM: reads Z2SSM data files (including the momentum dependence of the HZZ coupling), and determines Z2SSM benchmark points for HEPfit studies. Generates $\kappa_{Zh}^{240}$ vs. $\kappa_{Zh}^{365}$ plots. Uses data file with wrong implementation of the theoretical constraints, so a series of outlier points are seen in the plots which should have actually been discarded

* `Z2_SSM_scan_energy_dependence_plot.ipynb` $\rightarrow$ Shorter, cleaner version of `Z2_SSM_scan_energy_dependence.ipynb`, containing only the basic code to generate the $\kappa_{Zh}^{240}$ vs. $\kappa_{Zh}^{365}$ plots for the Z2SSM BPs. Uses data file with wrong implementation of the theoretical constraints, so a series of outlier points are seen in the plots which should have actually been discarded

* `SM_get_BRs.ipynb` $\rightarrow$ Short script to obtain the branching ratios of the SM Higgs boson, based on the SM higgstools example

### `setup_scripts/`

Contains the bash scripts to create the python virtual environment with all necessary libraries installed, including `higgstools\` (which should be stored in the main repository directory)

### `data/`

Contains the `.csv` datafiles for both models (Z2SSM and IDM).

### `plots/`

Directory to store the output plots

### `scan_output/`

Contains text files with the information on the resulting benchmark points (BPs) for each model. The contents of these files can be pasted into the python script used to scale the HEPfit observables according to the predictions of the benchmark points