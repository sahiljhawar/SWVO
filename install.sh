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
  echo "Environment name not provided, installing data_management in current environment"
else
  echo "Installing data_management as a package for environment: " "$PYTHON_ENV"
  source activate "$PYTHON_ENV" || conda activate "$PYTHON_ENV"
fi

echo
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

cd "$SCRIPT_DIR" || exit

pip install -r ./requirements.txt
python setup.py install
python setup.py develop
