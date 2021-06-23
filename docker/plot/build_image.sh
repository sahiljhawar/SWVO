CONTEXT=/PAGER/WP8/data_management/
IMAGE=wp8-plotting
DOCKERFILE=/PAGER/WP8/data_management/docker/plotting/Dockerfile

docker build -f $DOCKERFILE -t $IMAGE $CONTEXT