#!/bin/bash

cd /jwd
rm -rf higgs_venv

# module load anaconda/2023.09-py311  # No longer supported!! 
module load miniforge/24.7.1-0-py312

python -m venv higgs_venv
source ./higgs_venv/bin/activate
python -m pip install numpy scipy tqdm matplotlib pandas
python -m pip install ipykernel ipywidgets

cd /cephfs/user/mrebuzzi/phd/HiggsTools/future_projections/higgstools/
pip install .

cd /jwd
tar czf /cephfs/user/mrebuzzi/phd/HiggsTools/future_projections/setup_scripts/higgs_venv.tar.gz higgs_venv

cd /cephfs/user/mrebuzzi/phd/HiggsTools/future_projections/
