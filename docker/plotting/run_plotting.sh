INPUT_DIR=AGER/WP3/data/outputs/
CONTAINER_NAME=wp8_plotting_container
TEMP_RESULTS=/PAGER/WP8/data/temp/
IMAGE=wp8-plotting
FIGURES_DIR=/PAGER/WP3/data/figures/
LOG_DIR=/PAGER/WP3/data/logs/

docker run -v $INPUT_DIR:/inputs --name $CONTAINER_NAME  $IMAGE /bin/bash -c "bash /data_management/docker/plot_all_docker.sh"
mkdir -p $TEMP_RESULTS
docker container cp $CONTAINER_NAME:/results/. $TEMP_RESULTS
docker container cp $CONTAINER_NAME:/logs/. $TEMP_RESULTS
docker rm $CONTAINER_NAME

mv $TEMP_RESULTS/*.png $FIGURES_DIR
mv $TEMP_RESULTS/*.log $LOG_DIR