#!/usr/bin/env bash

Help() {
cat << EOF

   Usage: make_plots.sh [-I IMAGE] [-d DATA_FOLDER] [-r RECURRENT] [-U USERID] [-G GROUPID] [-P PLOT] [- l LOG_DIR]
                        [-D DATE]

   This script uses a docker container to generate a forecast using swift solar wind data as input. The machine learning
   model is build using the geoforecast library and saved.

   optional arguments:
       -I IMAGE        The name of the docker image to use
       -d DATA_FOLDER  Path to base pager data folder, input and output data are taken from there at the moment
       -r RECURRENT    To run it once or with sleeping time continuously. By default it runs once, otherwise set to
                       1 please
       -P PLOT         If specified, you perform only one specific plot. Now working "swift", "kp", "plasma" only
       -U USERID       The user id which writes files in the bind volume.
       -G GROUPID      The group id to which the written files belong to.
       -l LOG_DIR      (optional, not working properly)
       -D DATE         Date for which we want to plot the data in the format "YYYYMMDDHH" (e.g. 2021010110)

EOF
}

RECURRENT=0
PLOT=
LOG_DIR=
while getopts hI:d:r:P:U:G:l:D: flag
do
	case "${flag}" in
	        I) IMAGE=${OPTARG};;
	        d) DATA_FOLDER=${OPTARG};;
	        r) RECURRENT=${OPTARG};;
	        P) PLOT=${OPTARG};;
	        U) USERID=${OPTARG};;
	        G) GROUPID=${OPTARG};;
	        l) LOG_DIR=${OPTARG};;
	        D) DATE=${OPTARG};;
	        h) Help ;;
	        *) exit;;
    esac
done


if [ -z "$IMAGE" ]
then
  echo "You need to provide a name for a docker image(option -I). Exiting..."
  exit 1
fi

if [ -z "$DATA_FOLDER" ]
then
  DATA_FOLDER=/home/ruggero/PAGER/
fi

if [ -z "$USERID" ]
then
  USERID="65534"
fi

if [ -z "$GROUPID" ]
then
  GROUPID="65534"
fi

if [ -z "$DATE" ]
then
  echo "Date not specified, running real time..."
else

  if [[ "$DATE" =~ ^[0-2][0-9]{3}[0-1][0-9][0-3][0-9][0-2][0-9]$ ]]
  then
    echo
  else
    echo "Wrong date format, it should be a valid date in the format %YYYY%MM%DD%HH"
    exit 1
  fi
fi

if [ -z "$PLOT" ]
then
  echo "Plotting all products..."
else

  if [[ "$PLOT" == kp ]] || [[ "$PLOT" == swift ]] || [[ "$PLOT" == plasma ]]
  then
    echo "Plotting $PLOT data"
  else
    echo "Wrong plotting option, either leave blank or choose among kp, plasma or swift"
    exit 1
  fi
fi

docker run -d --rm -v "$DATA_FOLDER":/PAGER -u=$USERID:$GROUPID --env RECURRENT="$RECURRENT" --env LOG_DIR="$LOG_DIR" \
       --env DATE="$DATE" --env PLOT="$PLOT" \
       --entrypoint="./plot_entrypoint.sh" "$IMAGE"
