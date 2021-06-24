IMAGE=wp8-filecheck
INPUT_DIR=/PAGER/
PYTHON=/opt/conda/envs/pager/bin/python
TEMP_LOG_DIR=/logs/

#WP2
CONTAINER_NAME=wp8_filecheck_container
LOG_DIR=/PAGER/WP8/logs/

docker run -v $INPUT_DIR:$INPUT_DIR --network="host" --name $CONTAINER_NAME  $IMAGE /bin/bash -c "$PYTHON /data_management/scripts/check_file_generation/check_all_outputs.py -logdir $TEMP_LOG_DIR -notify 1"
docker container cp $CONTAINER_NAME:$TEMP_LOG_DIR. $LOG_DIR
docker rm $CONTAINER_NAME
