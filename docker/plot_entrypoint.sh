#!/usr/bin/env bash

pargs=()
pargs+=("-log" "$LOG_DIR");
pargs+=("-recurrent" "$RECURRENT");
pargs+=("-sleep" "$SLEEP_TIME_MINUTES");

if [ -z "$PLOT" ]
then
      /opt/conda/envs/"$PYTHON_ENV"/bin/python /scripts/plot/wp2/plot_all_swift.py "${pargs[@]}" \
      -input "$WP2_OUTPUT_FOLDER" \
      -output "$WP2_FIGURE_FOLDER"

      /opt/conda/envs/"$PYTHON_ENV"/bin/python /scripts/plot/wp3/plot_all_kp.py "${pargs[@]}"
      /opt/conda/envs/"$PYTHON_ENV"/bin/python /scripts/plot/wp3/plot_all_plasma.py "${pargs[@]}"
else
      case $PLOT in
        swift)
        /opt/conda/envs/"$PYTHON_ENV"/bin/python /scripts/plot/wp2/plot_all_swift.py  "${pargs[@]}" \
        -input "$WP2_OUTPUT_FOLDER" \
        -output "$WP2_FIGURE_FOLDER" ;;

        kp)
        /opt/conda/envs/"$PYTHON_ENV"/bin/python /scripts/plot/wp3/plot_all_kp.py  "${pargs[@]}" ;;
        plasma)
        /opt/conda/envs/"$PYTHON_ENV"/bin/python /scripts/plot/wp3/plot_all_plasma.py "${pargs[@]}" ;;
      esac
fi

