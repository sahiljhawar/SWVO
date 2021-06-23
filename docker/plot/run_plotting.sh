IMAGE=wp8-plotting
INPUT_DIR=/PAGER/
PYTHON=/opt/conda/envs/pager/bin/python
TEMP_RESULTS_DIR=/results/
TEMP_LOG_DIR=/logs/

#WP2
CONTAINER_NAME=wp8_plotting-wp2
LOG_DIR=/PAGER/WP2/logs/
RESULTS_DIR=/PAGER/WP2/data/figures/

docker run -v $INPUT_DIR:$INPUT_DIR --name $CONTAINER_NAME  $IMAGE /bin/bash -c "$PYTHON /data_management/scripts/plot/wp2/plot_all_swift.py -logdir $TEMP_LOG_DIR -output $TEMP_RESULTS_DIR"
docker container cp $CONTAINER_NAME:$TEMP_RESULTS_DIR. $RESULTS_DIR
docker container cp $CONTAINER_NAME:$TEMP_LOG_DIR. $LOG_DIR
docker rm $CONTAINER_NAME

#WP3 Kp
CONTAINER_NAME=wp8_plotting-wp3-kp
LOG_DIR=/PAGER/WP3/logs/
RESULTS_DIR=/PAGER/WP3/data/figures/

docker run -v $INPUT_DIR:$INPUT_DIR --name $CONTAINER_NAME  $IMAGE /bin/bash -c "$PYTHON /data_management/scripts/plot/wp3/plot_all_kp.py -logdir $TEMP_LOG_DIR -output $TEMP_RESULTS_DIR"
docker container cp $CONTAINER_NAME:$TEMP_RESULTS_DIR. $RESULTS_DIR
docker container cp $CONTAINER_NAME:$TEMP_LOG_DIR. $LOG_DIR
docker rm $CONTAINER_NAME

#WP3 Plasma
CONTAINER_NAME=wp8_plotting-wp3-plasma
LOG_DIR=/PAGER/WP3/logs/
RESULTS_DIR=/PAGER/WP3/data/figures/plasma/

docker run -v $INPUT_DIR:$INPUT_DIR --name $CONTAINER_NAME  $IMAGE /bin/bash -c "$PYTHON /data_management/scripts/plot/wp3/plot_all_plasma.py -logdir $TEMP_LOG_DIR -output $TEMP_RESULTS_DIR"
docker container cp $CONTAINER_NAME:$TEMP_RESULTS_DIR. $RESULTS_DIR
docker container cp $CONTAINER_NAME:$TEMP_LOG_DIR. $LOG_DIR
docker rm $CONTAINER_NAME