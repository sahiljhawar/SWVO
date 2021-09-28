#!/usr/bin/env bash

echo Installing PAGER data_management library into system...
echo

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

source activate kp

cd $SCRIPT_DIR
python setup.py install
python setup.py develop
pip install -r ./requirements.txt

source deactivate