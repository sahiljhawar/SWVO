#!/usr/bin/env bash

Help() {
cat << EOF

   Usage: make_local_environment.sh [-e PYTHON_ENV]

   This script is used to prepare a local conda environment to run checks and plotting routines.

   optional arguments:
       -e PYTHON_ENV   The name of the python conda environment to create. If not specified the
                       default "data_management" will be used
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
    echo "Python environment name not specified using default name data_management "
    PYTHON_ENV=data_management
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

conda create -n "$PYTHON_ENV" python=3.8
source activate "$PYTHON_ENV" || conda activate "$PYTHON_ENV"

cd "$SCRIPT_DIR" || exit
bash ../install.sh