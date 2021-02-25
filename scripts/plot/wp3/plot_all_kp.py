import os
import argparse
import datetime as dt
import matplotlib.pyplot as plt
import logging
import sys

sys.path.append("/PAGER/WP8/data_management/io/wp3/")
from read_kp import KPReader

sys.path.append("/PAGER/WP8/data_management/plotting/wp3/kp/")
from plot_kp import PlotKpOutput

DATA_PATH = "/PAGER/WP3/data/outputs/"
# RESULTS_PATH = "/PAGER/WP3/data/figures/"
RESULTS_PATH = "/home/ruggero/temp/"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-date', action="store", default=None, type=str,
                        help="Requested date to plot in the format %YYYY-%mm-%dd")
    args = parser.parse_args()
    plotting_date = dt.datetime.strptime(args.date, "%Y-%m-%d")

    reader = KPReader()
    plotter = PlotKpOutput()

    try:
        data_niemegk, date = reader.read("niemegk", plotting_date)
        plotter.plot_output(data_niemegk)
        plt.savefig(os.path.join(RESULTS_PATH, "Niemegk_LAST.png"))
        plt.savefig(os.path.join(RESULTS_PATH, "Niemegk_{}.png".format(date.strftime("%Y%m%d"))))
    except TypeError:
        logging.error(
            "Data for Niemegk nowcast for date {} not found...impossible to produce data plot...".format(plotting_date))

    try:
        data_swpc, date = reader.read("swpc", plotting_date)
        plotter.plot_output(data_swpc)
        plt.savefig(os.path.join(RESULTS_PATH, "SWPC_LAST.png"))
        plt.savefig(os.path.join(RESULTS_PATH, "SWPC_{}.png".format(date.strftime("%Y%m%d"))))
    except TypeError:
        logging.error(
            "Data for SWPC forecast for date {} not found...impossible to produce data plot...".format(plotting_date))

#    data_l1, date = reader.read("l1", plotting_date)
#    plotter.plot_output(data_l1)
#    plt.savefig(os.path.join(RESULTS_PATH, "L1_LAST.png"))
#    plt.savefig(os.path.join(RESULTS_PATH, "L1_{}.png".format(date.strftime("%Y%m%d"))))

#    data_swift, date = reader.read("swift", plotting_date)
#    plotter.plot_output(data_swift)
#    plt.savefig(os.path.join(RESULTS_PATH, "SWIFTKP_LAST.png"))
#    plt.savefig(os.path.join(RESULTS_PATH, "SWIFTKP_{}.png".format(date.strftime("%Y%m%d"))))
