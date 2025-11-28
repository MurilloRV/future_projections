#!/bin/bash


local=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  -l <local>   Create python venv locally in the BUDDY directory, instead of in /jwd
  -h           Show this help message
EOF
}

OPTSTRING="lh"

while getopts ${OPTSTRING} opt; do
  case ${opt} in
    l) local="_local"; echo "Creating the python venv in the working directory." ;;
    h) usage; exit 0 ;;
    ?) echo "Invalid option: -${OPTARG}."; exit 1 ;;
  esac
done

working_dir="/cephfs/user/mrebuzzi/phd/HiggsTools/future_projections/"

if [ "$local" == "_local" ]; then
    cd "$working_dir"
else
    cd /jwd
fi

rm -rf higgs_venv_py38

module load miniforge/4.9.2-7-py38

python -m venv higgs_venv_py38
source ./higgs_venv_py38/bin/activate
python -m pip install numpy==1.24 # pyCollier seems to be incompatible with numpy 2.0+
python -m pip install scipy pandas tqdm matplotlib iminuit uproot
python -m pip install ipykernel ipywidgets

export CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" # issue with cmake and pyCollier
export CMAKE_ARGS="-DF2PY_EXECUTABLE=$(which f2py)"
export F2PY=$(which f2py)
python -m pip install pyCollier anyBSM

# cd /cephfs/user/mrebuzzi/phd/HiggsTools/future_projections/higgstools/
# pip install .

# cd /jwd
tar czf /cephfs/user/mrebuzzi/phd/HiggsTools/future_projections/setup_scripts/higgs_venv_py38${local}.tar.gz higgs_venv_py38

cd "$working_dir"
