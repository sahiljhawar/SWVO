#!/usr/bin/env bash

Help() {
cat << EOF

   Usage: make_check.sh [-I IMAGE] [-d DATA_FOLDER] [-r RECURRENT] [-U USERID] [-G GROUPID] [-C CHECK] [- l LOG_DIR]

   This script uses a docker container to generate a forecast using swift solar wind data as input. The machine learning
   model is build using the geoforecast library and saved.

   optional arguments:
       -I IMAGE        The name of the docker image to use
       -d DATA_FOLDER  Path to base pager data folder, input and output data are taken from there at the moment
       -r RECURRENT    To run it once or with sleeping time continuously. By default it runs once, otherwise set to
                       1 please
       -C CHECK         If specified, you perform only one specific plot (not working at the moment, all are produced)
       -U USERID       The user id which writes files in the bind volume.
       -G GROUPID      The group id to which the written files belong to.
       -l LOG_DIR      (optional)
EOF
}

RECURRENT=0
CHECK=
LOG_DIR=
while getopts hI:d:r:C:U:G:l: flag
do
	case "${flag}" in
	        I) IMAGE=${OPTARG};;
	        d) DATA_FOLDER=${OPTARG};;
	        r) RECURRENT=${OPTARG};;
	        C) CHECK=${OPTARG};;
	        U) USERID=${OPTARG};;
	        G) GROUPID=${OPTARG};;
	        l) LOG_DIR=${OPTARG};;
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

docker run -d --rm -v "$DATA_FOLDER":/PAGER --net=host -u=$USERID:$GROUPID --env RECURRENT="$RECURRENT" \
       --env LOG_DIR="$LOG_DIR" \
       --entrypoint="./check_entrypoint.sh" "$IMAGE"

