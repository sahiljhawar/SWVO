#!/usr/bin/env bash

Help() {
cat << EOF

   Usage: run_plotting.sh [-e PYTHON_ENV] [-l LOG_FILE] [-i INPUT] [-o OUTPUT]

   This script is used to run all plotting routines for PAGER project products

   optional arguments:
       -e PYTHON_ENV   The name of the python conda environment to use
       -l LOG_FILE     Path with name of a log file to use. Not implemented yet..."
       -i INPUT        Path to an output data folder
       -o OUTPUT       Path to an input data folder

EOF
}

pargs=()
while getopts he:l:o:i: flag
do
	case "${flag}" in
	        e) PYTHON_ENV=${OPTARG};;
	        l) pargs+=("-log" "${OPTARG}");;
	        o) pargs+=("-output" "${OPTARG}");;
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
source activate "$PYTHON_ENV"

python ../scripts/plot/wp2/plot_all_swift.py "${pargs[@]}" -recurrent 0
python ../scripts/plot/wp3/plot_all_kp.py "${pargs[@]}" -recurrent 0
python ../scripts/plot/wp3/plot_all_plasma.py "${pargs[@]}" -recurrent 0


