#!/usr/bin/env bash

if [ -z "$LOG_DIR" ]
then
      pargs=()
else
      pargs=("-logdir" "$LOG_DIR");
fi

/opt/conda/envs/data_management/bin/python /scripts/plot/wp2/plot_all_swift.py "${pargs[@]}"

/opt/conda/envs/data_management/bin/python /scripts/plot/wp3/plot_all_kp.py "${pargs[@]}"

/opt/conda/envs/data_management/bin/python /scripts/plot/wp3/plot_all_plasma.py "${pargs[@]}"