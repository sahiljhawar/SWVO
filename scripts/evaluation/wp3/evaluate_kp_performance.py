import os
import argparse
import datetime as dt
import matplotlib.pyplot as plt
import logging
import pandas as pd

from data_management.io.wp3.read_kp import KPReader
from data_management.plotting.wp3.kp.plot_kp import PlotKpOutput


def read_niemegk_data(start_date, final_date):
    reader = KPReader()
    i = 0
    data = []
    while True:
        data_niemegk, _ = reader.read("niemegk", start_date + dt.timedelta(days=2 + i))
        if data_niemegk is not None:
            if i > 0:
                data_niemegk = data_niemegk[data_niemegk.index > max(data[-1].index)]
            data.append(data_niemegk)
            if max(data_niemegk.index) > final_date:
                break
        i += 1
    data = pd.concat(data, axis=0)
    data = data[data.index <= final_date]
    data = data[data.index >= start_date]
    return data


def read_forecast_data(start_date, final_date, horizon, source, model_name=None):
    reader = KPReader()
    values = []
    dates = []
    i = 0
    while True:
        if start_date + dt.timedelta(hours=3 * i) > final_date:
            break
        data_forecast, _ = reader.read(source, start_date + dt.timedelta(hours=3 * i), model_name="KP-FULL-SW-PAGER")
        forecast_value = data_forecast[data_forecast.index == start_date +
                                       dt.timedelta(hours=3 * i + horizon)]["kp"].values[0]
        values.append(forecast_value)
        dates.append(start_date + dt.timedelta(hours=(3 * i + horizon)))
        i += 1
    return pd.DataFrame({"kp": values}, index=dates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-date', action="store", default=None, type=str,
                        help="Requested date to plot in the format %YYYY%mm%dd%HH")
    parser.add_argument('-output', action="store", default="/PAGER/WP3/data/figures/", type=str,
                        help="Path to a folder where to store the produced figures")
    parser.add_argument('-input', action="store", default="/PAGER/WP3/data/outputs/", type=str,
                        help="Path to a folder where the data to plot is stored...(be more precise)")
    parser.add_argument('-logdir', action="store", default=None, type=str,
                        help="Log directory if logging is to be enabled.")

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

    if args.logdir is not None:
        log_file = "wp3_plot_all_kp_{}.log".format(plotting_date.strftime("%Y%m%dT%H%M%S"))
        logging.basicConfig(filename=os.path.join(args.logdir, log_file), level=logging.INFO,
                            datefmt="%Y-%m-%d %H:%M:%S",
                            format="%(asctime)s;%(levelname)s;%(message)s")

    RESULTS_PATH = args.output
    DATA_PATH = args.input

    start = dt.datetime(2021, 11, 2)
    final = dt.datetime(2021, 11, 5)

    df_niemegk = read_niemegk_data(start, final)
    df_forecast = read_forecast_data(start, final, source="l1", horizon=0)

    plotter = PlotKpOutput()
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    PlotKpOutput._add_subplot(ax, data=df_niemegk[["kp"]], title=None, width=0.5,
                                   ylabel=r'${}$'.format("K_{p}"), align="center")
    PlotKpOutput._add_subplot(ax, data=df_forecast[["kp"]], title=None, width=0.5,
                                   ylabel=r'${}$'.format("K_{p}"), alpha=1.0)
    plt.savefig("./Horizon0_comparison.png")