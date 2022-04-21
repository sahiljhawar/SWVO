import os
import argparse
import datetime as dt
import matplotlib.pyplot as plt
import logging
import time
import glob

from data_management.io.wp2.read_swift import SwiftReader, SwiftEnsembleReader
from data_management.plotting.wp2.swift.plot_swift import PlotSWIFTOutput


def swift_ensemble_plot_exists(output_folder, date):
    logging.info("Checking if swift ensemble figure has already been produced...")
    output_file = "SWIFT_DEF_ENSEMBLE_GFZ_{}.png".format(date.strftime("%Y%m%d"))
    figure_list = glob.glob(os.path.join(output_folder, output_file))
    if len(figure_list) == 0:
        return False
    else:
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-date', action="store", default=None, type=str,
                        help="Requested date to plot in the format %YYYY%mm%dd or %YYYY%mm%dd%HH")
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
        # We are plotting the date before since SWIFT produces outputs with delayed date
        plotting_date = date_now - dt.timedelta(hours=24)
    else:
        try:
            plotting_date = dt.datetime.strptime(args.date, "%Y%m%d")
        except ValueError:
            try:
                plotting_date = dt.datetime.strptime(args.date, "%Y%m%d%H")
            except ValueError:
                msg = "Provided date {} not in correct format %Y%m%d nor %Y%m%d%H . Aborting...".format(args.date)
                logging.error(msg)
                raise RuntimeError(msg)

    plotting_date = plotting_date.replace(hour=0)

    if args.log is not None:
        log_file = "wp2_plot_all_swift_gfz_{}.log".format(plotting_date.strftime("%Y%m%dT%H%M%S"))
        logging.basicConfig(filename=os.path.join(args.log, log_file), level=logging.INFO,
                            datefmt="%Y-%m-%d %H:%M:%S",
                            format="%(asctime)s;%(levelname)s;%(message)s")

    recurrent = bool(args.recurrent)
    plotter = PlotSWIFTOutput()

    while True:
        if swift_ensemble_plot_exists(args.output, plotting_date):
            if recurrent and (args.date is None):
                logging.info("Figure files present already...sleeping {} minutes".format(args.sleep))
                time.sleep(60 * args.sleep)
            else:
                logging.info("Figure files present already. Service not required. Exiting...")
        else:
            try:
                reader = SwiftEnsembleReader(wp2_output_folder=os.path.join(args.input, "SWIFT_ENSEMBLE/"))
                logging.info("Reading DEF-based SWIFT Ensemble original output data file...")
                data_gsm, data_hgc = reader.read(plotting_date)
                for i, _ in enumerate(data_gsm):
                    data_gsm[i] = data_gsm[i][data_gsm[i].index >= plotting_date]
                logging.info("...Complete!!")
                logging.info("Plotting and saving SWIFT Ensemble data plot")
                plotter.plot_ensemble_output(data_gsm)
                if plotting_date >= date_now:
                    plt.savefig(os.path.join(args.output, "SWIFT_DEF_ENSEMBLE_GFZ_LAST.png"))
                plt.savefig(
                    os.path.join(args.output,
                                 "SWIFT_DEF_ENSEMBLE_GFZ_{}.png".format(plotting_date.strftime("%Y%m%d"))))
                logging.info("...Complete!!")
            except (TypeError, FileNotFoundError):
                logging.error(
                    "Data for DEF-based SWIFT Ensemble solar wind for date {} not found..."
                    "impossible to produce data plot...".format(plotting_date))

        if not recurrent:
            break

    #    try:
    #        reader = SwiftReader()
    #        logging.info("Reading AWSOM-based SWIFT original output data file...")
    #        data_gsm, data_hgc = reader.read(plotting_date)
    #        data_gsm = data_gsm[data_gsm.index >= plotting_date]
    #        logging.info("...Complete!!")
    #        logging.info("Plotting and saving SWIFT data plot")
    #        plotter.plot_output(data_gsm)
    #        if plotting_date >= date_now:
    #            plt.savefig(os.path.join(RESULTS_PATH, "SWIFT_GFZ_LAST.png"))
    #        plt.savefig(os.path.join(RESULTS_PATH, "SWIFT_GFZ_{}.png".format(plotting_date.strftime("%Y%m%d"))))
    #        logging.info("...Complete!!")
    #    except (TypeError, FileNotFoundError):
    #        logging.error(
    #            "Data for AWSOM-based SWIFT solar wind for date {} not found..."
    #            "impossible to produce data plot...".format(plotting_date))

    #    try:
    #        reader = SwiftReader(wp2_output_folder=os.path.join(args.input, "SWIFT_BIAS_CORRECTED/"))
    #        logging.info("Reading SWIFT data with bias correction file...")
    #        data_gsm, _ = reader.read(plotting_date, file_type="gsm")
    #        data_gsm = data_gsm[data_gsm.index >= plotting_date]
    #        data_gsm = data_gsm.resample("20T").pad()
    #        logging.info("...Complete!!")
    #        logging.info("Plotting and saving SWIFT bias corrected data plot")
    #        plotter.plot_output(data_gsm)
    #        if plotting_date >= date_now:
    #            plt.savefig(os.path.join(RESULTS_PATH, "SWIFT_GFZ_BIAS_LAST.png"))
    #        plt.savefig(os.path.join(RESULTS_PATH, "SWIFT_GFZ_BIAS_{}.png".format(plotting_date.strftime("%Y%m%d"))))
    #        logging.info("...Complete!!")
    #    except (TypeError, FileNotFoundError):
    #        logging.error(
    #            "Data for SWIFT solar wind with bias correction for date {} not found..."
    #            "impossible to produce data plot...".format(plotting_date))

    #    try:
    #        reader = SwiftReader(wp2_output_folder=os.path.join(args.input, "SWIFT_DEF/"))
    #        logging.info("Reading DEF-based SWIFT original output data file...")
    #        data_gsm, data_hgc = reader.read(plotting_date)
    #        data_gsm = data_gsm[data_gsm.index >= plotting_date]
    #        logging.info("...Complete!!")
    #        logging.info("Plotting and saving SWIFT data plot")
    #        plotter.plot_output(data_gsm)
    #        if plotting_date >= date_now:
    #            plt.savefig(os.path.join(RESULTS_PATH, "SWIFT_DEF_GFZ_LAST.png"))
    #        plt.savefig(os.path.join(RESULTS_PATH, "SWIFT_DEF_GFZ_{}.png".format(plotting_date.strftime("%Y%m%d"))))
    #        logging.info("...Complete!!")
    #    except (TypeError, FileNotFoundError):
    #        logging.error(
    #            "Data for DEF-based SWIFT solar wind for date {} not found..."
    #            "impossible to produce data plot...".format(plotting_date))
