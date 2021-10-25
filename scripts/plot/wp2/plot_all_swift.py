import os
import argparse
import datetime as dt
import matplotlib.pyplot as plt
import logging

from data_management.io.wp2.read_swift import SwiftReader
from data_management.plotting.wp2.swift.plot_swift import PlotSWIFTOutput

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-date', action="store", default=None, type=str,
                        help="Requested date to plot in the format %YYYY%mm%dd")
    parser.add_argument('-output', action="store", default="/PAGER/WP3/data/figures/", type=str,
                        help="Path to a folder where to store the produced figures")
    parser.add_argument('-input', action="store", default="/PAGER/WP2/data/outputs/", type=str,
                        help="Path to a folder where the data to plot is stored...(be more precise)")
    parser.add_argument('-logdir', action="store", default=None, type=str,
                        help="Log directory if logging is to be enabled.")

    args = parser.parse_args()

    date_now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    if args.date is None:
        plotting_date = date_now
    else:
        try:
            plotting_date = dt.datetime.strptime(args.date, "%Y%m%d")
        except TypeError:
            msg = "Provided date {} not in correct format %Y%m%d. Aborting...".format(args.date)
            logging.error(msg)
            raise RuntimeError(msg)

    if args.logdir is not None:
        log_file = "wp2_plot_all_swift_gfz_{}.log".format(plotting_date.strftime("%Y%m%dT%H%M%S"))
        logging.basicConfig(filename=os.path.join(args.logdir, log_file), level=logging.INFO,
                            datefmt="%Y-%m-%d %H:%M:%S",
                            format="%(asctime)s;%(levelname)s;%(message)s")

    RESULTS_PATH = args.output
    plotter = PlotSWIFTOutput()

    try:
        reader = SwiftReader()
        logging.info("Reading AWSOM-based SWIFT original output data file...")
        data_gsm, data_hgc = reader.read(plotting_date)
        data_gsm = data_gsm[data_gsm.index >= plotting_date]
        logging.info("...Complete!!")
        logging.info("Plotting and saving SWIFT data plot")
        plotter.plot_output(data_gsm)
        if plotting_date >= date_now:
            plt.savefig(os.path.join(RESULTS_PATH, "SWIFT_GFZ_LAST.png"))
        plt.savefig(os.path.join(RESULTS_PATH, "SWIFT_GFZ_{}.png".format(plotting_date.strftime("%Y%m%d"))))
        logging.info("...Complete!!")
    except (TypeError, FileNotFoundError):
        logging.error(
            "Data for AWSOM-based SWIFT solar wind for date {} not found..."
            "impossible to produce data plot...".format(plotting_date))

    try:
        reader = SwiftReader(wp2_output_folder=os.path.join(args.input, "SWIFT_BIAS_CORRECTED/"))
        logging.info("Reading SWIFT data with bias correction file...")
        data_gsm, _ = reader.read(plotting_date, file_type="gsm")
        data_gsm = data_gsm[data_gsm.index >= plotting_date]
        data_gsm = data_gsm.resample("20T").pad()
        logging.info("...Complete!!")
        logging.info("Plotting and saving SWIFT bias corrected data plot")
        plotter.plot_output(data_gsm)
        if plotting_date >= date_now:
            plt.savefig(os.path.join(RESULTS_PATH, "SWIFT_GFZ_BIAS_LAST.png"))
        plt.savefig(os.path.join(RESULTS_PATH, "SWIFT_GFZ_BIAS_{}.png".format(plotting_date.strftime("%Y%m%d"))))
        logging.info("...Complete!!")
    except (TypeError, FileNotFoundError):
        logging.error(
            "Data for SWIFT solar wind with bias correction for date {} not found..."
            "impossible to produce data plot...".format(plotting_date))

    try:
        reader = SwiftReader(wp2_output_folder=os.path.join(args.input, "SWIFT_DEF/"))
        logging.info("Reading DEF-based SWIFT original output data file...")
        data_gsm, data_hgc = reader.read(plotting_date)
        data_gsm = data_gsm[data_gsm.index >= plotting_date]
        logging.info("...Complete!!")
        logging.info("Plotting and saving SWIFT data plot")
        plotter.plot_output(data_gsm)
        if plotting_date >= date_now:
            plt.savefig(os.path.join(RESULTS_PATH, "SWIFT_DEF_GFZ_LAST.png"))
        plt.savefig(os.path.join(RESULTS_PATH, "SWIFT_DEF_GFZ_{}.png".format(plotting_date.strftime("%Y%m%d"))))
        logging.info("...Complete!!")
    except (TypeError, FileNotFoundError):
        logging.error(
            "Data for DEF-based SWIFT solar wind for date {} not found..."
            "impossible to produce data plot...".format(plotting_date))
