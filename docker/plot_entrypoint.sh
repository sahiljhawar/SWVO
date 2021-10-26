#!/usr/bin/env bash

/opt/conda/envs/data_management/bin/python /scripts/plot/wp2/plot_all_swift.py -logdir /PAGER/WP2/logs/

/opt/conda/envs/data_management/bin/python /scripts/plot/wp3/plot_all_kp.py -logdir /PAGER/WP3/logs/

/opt/conda/envs/data_management/bin/python /scripts/plot/wp3/plot_all_plasma.py -logdir /PAGER/WP3/logs/