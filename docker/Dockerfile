FROM continuumio/miniconda:latest

LABEL maintainer="ruggero.vasile@gfz-potsdam.de"
LABEL version="beta"
LABEL description="Docker image to run routines for data management of PAGER project"

SHELL ["/bin/bash", "-c"]

RUN mkdir data_management
RUN mkdir data_management/data_management/
COPY ./data_management/ data_management/data_management/
COPY ./install.sh data_management/
COPY ./setup.py data_management/
COPY ./requirements.txt data_management/

RUN mkdir scripts/
RUN mkdir scripts/plot
RUN mkdir scripts/check_file_generation
COPY ./scripts/plot/ /scripts/plot/
COPY ./scripts/check_file_generation/ /scripts/check_file_generation/
COPY ./docker/plot_entrypoint.sh /

RUN conda create --name data_management python=3.8
SHELL ["conda", "run", "-n", "data_management", "/bin/bash", "-c"]

RUN bash data_management/install.sh

RUN apt-get update --allow-releaseinfo-change
RUN apt-get update --fix-missing -y
RUN apt-get install -y ffmpeg
