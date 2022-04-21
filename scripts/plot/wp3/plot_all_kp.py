import os
import argparse
import datetime as dt
import matplotlib.pyplot as plt
import logging

from data_management.io.wp3.read_kp import KPReader
from data_management.plotting.wp3.kp.plot_kp import PlotKpOutput

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-date', action="store", default=None, type=str,
                        help="Requested date to plot in the format %YYYY%mm%dd%HH")
    parser.add_argument('-output', action="store", default=None, type=str,
                        help="Path to a folder where to store the produced figures")
    parser.add_argument('-input', action="store", default=None, type=str,
                        help="Path to a folder where the data to plot is stored...(be more precise)")
    parser.add_argument('-log', action="store", default=None, type=str,
                        help="Log directory if logging is to be enabled.")
    parser.add_argument('-recurrent', action="store", default=None, type=int,
                        help="True if you want to keep running, False if you want to run it only once")
    parser.add_argument('-sleep', action="store", default=None, type=int,
                        help="Time for the script to sleep in minutes")

    args = parser.parse_args()

    date_now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    if args.date is None:
        plotting_date = date_now
    else:
        try:
            plotting_date = dt.datetime.strptime(args.date, "%Y%m%d%H")
        except ValueError:
            msg = "Provided date {} not in correct format %Y%m%d%H. Aborting...".format(args.date)
            logging.error(msg)
            raise RuntimeError(msg)

    if args.log is not None:
        log_file = "wp3_plot_all_kp_{}.log".format(plotting_date.strftime("%Y%m%dT%H%M%S"))
        logging.basicConfig(filename=os.path.join(args.log, log_file), level=logging.INFO,
                            datefmt="%Y-%m-%d %H:%M:%S",
                            format="%(asctime)s;%(levelname)s;%(message)s")

    RESULTS_PATH = args.output
    DATA_PATH = args.input

    reader = KPReader()
    plotter = PlotKpOutput()

    try:
        logging.info("Reading Niemegk Kp nowcast data file...")
        data_niemegk, date = reader.read("niemegk", plotting_date)
        logging.info("...Complete!!")
        logging.info("Plotting and saving Niemegk Kp nowcast data...")
        plotter.plot_output(data_niemegk)
        if plotting_date >= date_now:
            plt.savefig(os.path.join(RESULTS_PATH, "Niemegk_LAST.png"))
        plt.savefig(os.path.join(RESULTS_PATH, "Niemegk_{}.png".format(date.strftime("%Y%m%d"))))
        logging.info("...Complete!!")
    except TypeError:
        logging.error(
            "Data for Niemegk nowcast for date {} not found...impossible to produce data plot...".format(plotting_date))
    
    try:
        logging.info("Reading SWPC Kp forecast data file...")
        data_swpc, date = reader.read("swpc", plotting_date)
        logging.info("...Complete!!")
        logging.info("Plotting and saving SWPC Kp nowcast data...")
        plotter.plot_output(data_swpc)
        if plotting_date >= date_now:
            plt.savefig(os.path.join(RESULTS_PATH, "SWPC_LAST.png"))
        plt.savefig(os.path.join(RESULTS_PATH, "SWPC_{}.png".format(date.strftime("%Y%m%d"))))
        logging.info("...Complete!!")
    except TypeError:
        logging.error(
            "Data for SWPC forecast for date {} not found...impossible to produce data plot...".format(plotting_date))

    for model_name in ["KP-FULL-SW-PAGER", "HP60-FULL-SW-SWAMI-PAGER"]:
        try:
            logging.info("Reading L1 Kp forecast data file for model {}...".format(model_name))
            data_l1, date = reader.read("l1", plotting_date, model_name=model_name)
            logging.info("...Complete!!")
            logging.info("Plotting and saving L1 Kp forecast data for model {}...".format(model_name))
            plotter.plot_output(data_l1)
            if plotting_date >= date_now:
                plt.savefig(os.path.join(RESULTS_PATH, "L1_LAST_{}.png".format(model_name)))
            plt.savefig(os.path.join(RESULTS_PATH, "L1_{}_{}.png".format(date.strftime("%Y%m%dT%H%M%S"), model_name)))
            logging.info("...Complete!!")
        except TypeError:
            logging.error(
                "Data for L1 Kp forecast for date {} and model {} not found...impossible to produce data plot...".format(plotting_date, model_name))


#    try:
#        logging.info("Reading SWIFT Kp forecast data file...")
#        data_swift, date = reader.read("swift", plotting_date)
#        logging.info("...Complete!!")
#        logging.info("Plotting and saving SWIFT Kp forecast data...")
#        plotter.plot_output(data_swift)
#        if plotting_date >= date_now:
#            plt.savefig(os.path.join(RESULTS_PATH, "SWIFT_LAST.png"))
#        plt.savefig(os.path.join(RESULTS_PATH, "SWIFT_{}.png".format(date.strftime("%Y%m%dT%H%M%S"))))
#        logging.info("...Complete!!")
#    except TypeError:
#        logging.error(
#            "Data for SWIFT Kp forecast for date {} not found...impossible to produce data plot...".format(
#                plotting_date))
