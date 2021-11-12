#!/usr/bin/env bash

Help() {
cat << EOF

   Usage: run_plotting.sh [-e PYTHON_ENV] [-l LOG_FILE] [-i INPUT] [-o OUTPUT] [-D DATE]

   This script is used to run all plotting routines for PAGER project products

   optional arguments:
       -e PYTHON_ENV   The name of the python conda environment to use
       -l LOG_FILE     Path with name of a log file to use. Not implemented yet..."
       -i INPUT        Path to an output data folder
       -o OUTPUT       Path to an input data folder
       -D DATE         Date for which we want to plot the data in the format "YYYYMMDDHH" (e.g. 2021010110)

EOF
}

pargs=()
while getopts he:l:o:i:D: flag
do
	case "${flag}" in
	        e) PYTHON_ENV=${OPTARG};;
	        l) pargs+=("-logdir" "${OPTARG}");;
	        o) pargs+=("-output" "${OPTARG}");;
	        i) pargs+=("-input" "${OPTARG}");;
	        D) DATE=${OPTARG};;
	        #D) pargs+=("-date" "${OPTARG}");;
	        h) Help ;;
	        *) exit;;
    esac
done


if [ -z "$PYTHON_ENV" ]
then
  echo "You need to provide a name for a python environment (option -e). Exiting..."
  exit 1
fi

if [ -z "$DATE" ]
then
  echo "Date not specified, running real time..."
else

  if [[ "$DATE" =~ ^[0-2][0-9]{3}[0-1][0-9][0-3][0-9][0-2][0-9]$ ]]
  then
    pargs+=("-date" "${DATE}");
  else
    echo "Wrong date format, it should be a valid date in the format %YYYY%MM%DD%HH"
    exit 1
  fi
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

cd "$SCRIPT_DIR" || exit
source activate "$PYTHON_ENV" || conda activate "$PYTHON_ENV"

python ../scripts/plot/wp2/plot_all_swift.py "${pargs[@]}"
python ../scripts/plot/wp3/plot_all_kp.py "${pargs[@]}"
python ../scripts/plot/wp3/plot_all_plasma.py "${pargs[@]}"


