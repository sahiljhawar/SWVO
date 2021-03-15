PYTHON=/opt/conda/envs/pager/bin/python

mkdir -p /logs
mkdir -p /results

$PYTHON /data_management/scripts/plot/wp3/plot_all_kp.py -input /inputs/ -output /results -logdir /logs