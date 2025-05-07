#!/bin/bash

cd /jwd
rm -rf higgs_venv

# module load anaconda/2023.09-py311  # No longer supported!! 
module load miniforge/24.7.1-0-py312

tar xf /cephfs/user/mrebuzzi/phd/HiggsTools/future_projections/setup_scripts/higgs_venv.tar.gz
source ./higgs_venv/bin/activate

cd /cephfs/user/mrebuzzi/phd/HiggsTools/future_projections/
