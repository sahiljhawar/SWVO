#!/usr/bin/env bash

Help() {
cat << EOF

   Usage: run_checks.sh [-e PYTHON_ENV] [-l LOG_FILE] [-i INPUT] [-n NOTIFY]

   This script is used to run all plotting routines for PAGER project products

   optional arguments:
       -e PYTHON_ENV   The name of the python conda environment to use
       -l LOG_FILE     Path with name of a log file to use. Not implemented yet..."
       -i INPUT        Path to an output data folder
       -n NOTIFY

EOF
}

pargs=()
NOTIFY=0
while getopts he:l:n:i: flag
do
	case "${flag}" in
	        e) PYTHON_ENV=${OPTARG};;
	        l) pargs+=("-logdir" "${OPTARG}");;
	        n) NOTIFY=${OPTARG};;
	        i) pargs+=("-input" "${OPTARG}");;
	        h) Help ;;
	        *) exit;;
    esac
done


if [ -z "$PYTHON_ENV" ]
then
  echo "You need to provide a name for a python environment (option -e). Exiting..."
  exit 1
fi


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

cd "$SCRIPT_DIR" || exit
source activate "$PYTHON_ENV" || conda activate "$PYTHON_ENV"

echo "${pargs[@]}"

python ../scripts/check_file_generation/check_all_outputs.py "${pargs[@]}" -notify "$NOTIFY"



