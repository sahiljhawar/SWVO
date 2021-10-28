#!/usr/bin/env bash

if [ -z "$LOG_DIR" ]
then
      pargs=()
else
      pargs=("-logdir" "$LOG_DIR");
fi

/opt/conda/envs/data_management/bin/python /scripts/check_file_generation/check_all_outputs.py "${pargs[@]}" \
                                           -notify "$NOTIFY"