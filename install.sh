#!/usr/bin/env bash

Help() {
cat << EOF

   Usage: make_local_environment.sh [-e PYTHON_ENV]

   This script is used to install a the geoforecast library.

   optional arguments:
       -e PYTHON_ENV   The name of the python conda environment to use.
EOF
}

while getopts he: flag
do
	case "${flag}" in
	        e) PYTHON_ENV=${OPTARG};;
	        h) Help ;;
	        *) exit;;
    esac
done

if [ -z "$PYTHON_ENV" ]
then
  echo "You need to provide a name for a python environment (option -e). Exiting..."
  exit 1
fi

echo Installing PAGER data_management library into system into existing environment "$PYTHON_ENV"
echo

source activate "$PYTHON_ENV" || conda activate "$PYTHON_ENV"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

cd "$SCRIPT_DIR" || exit
python setup.py install
python setup.py develop
pip install -r ./requirements.txt
