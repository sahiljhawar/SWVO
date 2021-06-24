FROM continuumio/miniconda

LABEL maintainer="ruggero.vasile@gfz-potsdam.de"
LABEL version="beta"
LABEL description="Docker image to run plotting routines for PAGER project"

RUN mkdir data_management
RUN mkdir data_management/data_management/
RUN mkdir data_management/data_management/io/
COPY data_management/io/ /data_management/data_management/io/

RUN mkdir data_management/data_management/plotting/
COPY data_management/plotting/ /data_management/data_management/plotting/

RUN mkdir data_management/scripts/
RUN mkdir data_management/scripts/plot/
COPY scripts/plot/ /data_management/scripts/plot/

RUN mkdir logs/
RUN mkdir results/

RUN conda create --name pager python=3.8
RUN /opt/conda/envs/pager/bin/pip install -r /data_management/scripts/plot/requirements.txt
RUN apt-get update --fix-missing 
RUN apt-get install ffmpeg
