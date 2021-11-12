#!/usr/bin/env bash

if [ -z "$LOG_DIR" ]
then
      pargs=()
else
      pargs=("-logdir" "$LOG_DIR");
fi

if [ -z "$DATE" ]
then
      echo
else
      pargs=("-date" "$DATE");
fi

if [ -z "$PLOT" ]
then
      /opt/conda/envs/data_management/bin/python /scripts/plot/wp2/plot_all_swift.py "${pargs[@]}"
      /opt/conda/envs/data_management/bin/python /scripts/plot/wp3/plot_all_kp.py "${pargs[@]}"
      /opt/conda/envs/data_management/bin/python /scripts/plot/wp3/plot_all_plasma.py "${pargs[@]}"
else
      case $PLOT in
        swift)
        /opt/conda/envs/data_management/bin/python /scripts/plot/wp2/plot_all_swift.py "${pargs[@]}" ;;
        kp)
        /opt/conda/envs/data_management/bin/python /scripts/plot/wp3/plot_all_kp.py "${pargs[@]}" ;;
        plasma)
        /opt/conda/envs/data_management/bin/python /scripts/plot/wp3/plot_all_plasma.py "${pargs[@]}" ;;
      esac
fi

