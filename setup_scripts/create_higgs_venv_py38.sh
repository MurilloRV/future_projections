#!/bin/bash

cd /jwd
rm -rf higgs_venv_py38

module load miniforge/4.9.2-7-py38

python -m venv higgs_venv_py38
source ./higgs_venv_py38/bin/activate
python -m pip install numpy==1.24 # pyCollier seems to be incompatible with numpy 2.0+
python -m pip install scipy pandas tqdm matplotlib
python -m pip install ipykernel ipywidgets

# export CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" # issue with cmake and pyCollier
python -m pip install pyCollier

# cd /cephfs/user/mrebuzzi/phd/HiggsTools/future_projections/higgstools/
# pip install .

cd /jwd
tar czf /cephfs/user/mrebuzzi/phd/HiggsTools/future_projections/setup_scripts/higgs_venv_py38.tar.gz higgs_venv_py38

cd /cephfs/user/mrebuzzi/phd/HiggsTools/future_projections/
