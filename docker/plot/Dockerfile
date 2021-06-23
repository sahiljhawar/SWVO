FROM continuumio/miniconda

LABEL maintainer="ruggero.vasile@gfz-potsdam.de"
LABEL version="beta"
LABEL description="Docker image to run plotting routines for PAGER project"

RUN mkdir data_management

RUN mkdir data_management/io/
COPY ./io/ /data_management/io/

RUN mkdir data_management/plotting/
COPY ./plotting/ /data_management/plotting/

RUN mkdir data_management/docker/
COPY ./docker/plot_all_docker.sh /data_management/docker/

RUN mkdir data_management/scripts/
RUN mkdir data_management/scripts/plot/
COPY ./scripts/plot/ /data_management/scripts/plot/

RUN conda create --name pager python=3.8
RUN /opt/conda/envs/pager/bin/pip install -r /data_management/scripts/plot/requirements.txt