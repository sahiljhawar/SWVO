FROM continuumio/miniconda

LABEL maintainer="ruggero.vasile@gfz-potsdam.de"
LABEL version="beta"
LABEL description="Docker image to run output file check routines for PAGER project"

RUN mkdir data_management
RUN mkdir data_management/data_management/

RUN mkdir data_management/data_management/io/
COPY data_management/io/ /data_management/data_management/io/

RUN mkdir data_management/data_management/notifications/
COPY data_management/notifications/ /data_management/data_management/notifications/

RUN mkdir data_management/data_management/check/
COPY data_management/check/ /data_management/data_management/check/

RUN mkdir data_management/scripts/
RUN mkdir data_management/scripts/check_file_generation/
COPY scripts/check_file_generation/ /data_management/scripts/check_file_generation/

RUN mkdir logs/

RUN conda create --name pager python=3.8
RUN /opt/conda/envs/pager/bin/pip install -r /data_management/scripts/check_file_generation/requirements.txt
