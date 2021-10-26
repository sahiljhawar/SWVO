#!/usr/bin/env bash

# TODO Add a tag option

Help() {
cat << EOF

   Usage: generate_forecast.sh [-I IMAGE] [-R REGISTRY]

   This script builds a docker image using instructions in the Dockerfile of the same folder. If a valid
   registry is provided, after the image is locally build will be also uploaded to the registry.

   optional arguments:
       -I IMAGE        (Optional) The name of the docker image to build. If not provided a default name
                                  will be used.
       -R REGISTRY     (Optional) The image can be uploaded to a docker registry if requested
EOF
}

while getopts hI:R: flag
do
	case "${flag}" in
	        I) IMAGE=${OPTARG};;
	        R) REGISTRY=${OPTARG};;
	        h) Help ;;
	        *) exit;;
    esac
done

if [ -z "$IMAGE" ]
then
  IMAGE=data-management
  echo "Using default image name"
  echo $IMAGE
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
docker build --no-cache -t $IMAGE -f "$SCRIPT_DIR"/Dockerfile "$SCRIPT_DIR"/../

if [ -z "$REGISTRY" ]
then
  echo "Registry not provided. Not uploading image to any registry."
else
  echo "Uploading Image to Registry:", "$REGISTRY"
  docker tag $IMAGE "$REGISTRY"/"$IMAGE"
  docker push "$REGISTRY"/"$IMAGE" || exit
  docker image rm $IMAGE
fi