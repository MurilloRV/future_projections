#!/bin/bash
source /etc/profile
source $BUDDY/.bashrc_Rocky9

local=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  -l <local>   Unpack the python venv locally in the BUDDY directory, instead of in /jwd
  -h           Show this help message
EOF
}

OPTSTRING=":lh"

while getopts ${OPTSTRING} opt; do
  case ${opt} in
    l) local="_local"; echo "Unpacking the python venv in the working directory." ;;
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

tar xf "${working_dir}setup_scripts/higgs_venv_py38${local}.tar.gz"
source ./higgs_venv_py38/bin/activate

cd "$working_dir"

git config --global user.name "Murillo Vellasco"
git config --global user.email "murillovellasco@gmail.com"