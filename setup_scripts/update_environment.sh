#!/bin/bash

# pip install ./higgstools
pip install --upgrade --no-deps --force-reinstall ./higgstools
cd
tar czf /cephfs/user/mrebuzzi/phd/HiggsTools/future_projections/setup_scripts/higgs_venv.tar.gz higgs_venv
cd /cephfs/user/mrebuzzi/phd/HiggsTools/future_projections/
