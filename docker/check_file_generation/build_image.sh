CONTEXT=/PAGER/WP8/data_management/
IMAGE=wp8-filecheck
DOCKERFILE=/PAGER/WP8/data_management/docker/check_file_generation/Dockerfile

docker build -f $DOCKERFILE -t $IMAGE $CONTEXT
