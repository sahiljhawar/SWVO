#!/usr/bin/env bash

ENV_NAME=data_management
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

conda create -n "$ENV_NAME" python=3.8
source activate "$ENV_NAME"

cd "$SCRIPT_DIR" || exit
bash ../install.sh