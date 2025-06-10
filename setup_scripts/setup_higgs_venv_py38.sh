#!/bin/bash

cd /jwd
rm -rf higgs_venv_py38

module load miniforge/4.9.2-7-py38

tar xf /cephfs/user/mrebuzzi/phd/HiggsTools/future_projections/setup_scripts/higgs_venv_py38.tar.gz
source ./higgs_venv_py38/bin/activate

cd /cephfs/user/mrebuzzi/phd/HiggsTools/future_projections/

git config --global user.name "Murillo Vellasco"
git config --global user.email "murillovellasco@gmail.com"