import os
import argparse
import datetime as dt
import logging

import sys
sys.path.append("/PAGER/WP8/data_management/")

from data_management.io.wp3.read_plasmasphere import PlasmaspherePredictionReader
from data_management.plotting.wp3.plasmasphere.plasmasphere_plot import PlasmaspherePlot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-date', action="store", default=None, type=str,
                        help="Requested date to plot in the format %YYYY-%mm-%dd")
    parser.add_argument('-output', action="store", default="/PAGER/WP3/data/figures/plasmasphere/", type=str,
                        help="Path to a folder where to store the produced figures")
    parser.add_argument('-input', action="store", default="/PAGER/WP3/data/outputs/", type=str,
                        help="Path to a folder where the data to plot is stored...(be more precise)")
    parser.add_argument('-logdir', action="store", default="/PAGER/WP3/logs/plasmasphere/", type=str,
                        help="Log directory if logging is to be enabled.")

    args = parser.parse_args()

    date_now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    if args.date is None:
        plotting_date = date_now
    else:
        try:
            plotting_date = dt.datetime.strptime(args.date, "%Y-%m-%d-%H-%M")
        except TypeError:
            msg = "Provided date {} not in correct format %Y-%m-%d. Aborting...".format(args.date)
            logging.error(msg)
            raise RuntimeError(msg)

    if args.logdir is not None:
        log_file = "wp3_plot_all_plasma_{}.log".format(plotting_date.strftime("%Y%m%dT%H%M%S"))
        logging.basicConfig(filename=os.path.join(args.logdir, log_file), level=logging.INFO)

    RESULTS_PATH = args.output
    DATA_PATH = args.input

    reader = PlasmaspherePredictionReader(wp3_output_folder=DATA_PATH)
    plotter = PlasmaspherePlot()

    video_name = "gfz_plasma_video_{}.mp4".format(plotting_date.strftime("%Y%m%dT%H%M"))
    try:
        logging.info("Reading GFZ Plasmasphere forecast data file...")
        data = reader.read("gfz_plasma", plotting_date)
        logging.info("...Complete!!")
        logging.info("Plotting and saving plasmasphere forecast video...")
        plotter.plot_output(data, RESULTS_PATH, video_name)
        logging.info("...Complete!!")
    except TypeError:
        logging.error(
            "Data for Plasmasphere forecast for date {} not found"
            "...impossible to produce data plot...".format(plotting_date)
        )