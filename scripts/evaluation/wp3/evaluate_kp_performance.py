import os
import argparse
import datetime as dt
import matplotlib.pyplot as plt
import logging
import pandas as pd
import matplotlib.patches as patches

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


def read_forecast_data(start_date, final_date, horizon, source, model_name="KP-FULL-SW-PAGER"):
    reader = KPReader()
    values = []
    dates = []
    i = 0
    while True:
        if start_date + dt.timedelta(hours=3 * i) > final_date:
            break
        data_forecast, _ = reader.read(source, start_date + dt.timedelta(hours=3 * i - horizon), model_name=model_name)
        forecast_value = data_forecast[data_forecast.index == start_date +
                                       dt.timedelta(hours=3 * i)]["kp"].values[0]
        values.append(forecast_value)
        dates.append(start_date + dt.timedelta(hours=(3 * i)))
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
    horizon = 6
    df_forecast = read_forecast_data(start, final, source="l1", horizon=horizon)

    df_niemegk = df_niemegk[df_niemegk.index >= min(df_forecast.index)]
    df_niemegk = df_niemegk[df_niemegk.index <= max(df_forecast.index)]
    df_forecast = df_forecast[df_forecast.index >= min(df_niemegk.index)]
    df_forecast = df_forecast[df_forecast.index <= max(df_niemegk.index)]

    plotter = PlotKpOutput()
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0.08, bottom=0.15, right=0.95, top=0.9, wspace=None, hspace=0.6)

    green_patch = patches.Patch(color=[0.0, 0.5, 0.0, 1.0], label="Kp Nowcast")
    red_patch = patches.Patch(color=[0.5, 0.0, 0.0, 1.0], label="Kp Forecast")

    ax.legend(bbox_to_anchor=(0., 1., 0.84, .275),
              handles=[green_patch, red_patch],
              ncol=2, fontsize="xx-large", shadow=True)

    color_niemegk = []
    for i in range(len(df_niemegk)):
        color_niemegk.append([0.0, 0.5, 0.0, 1.0])

    color_forecast = []
    for i in range(len(df_forecast)):
        color_forecast.append([0.5, 0.0, 0.0, 1.0])

    PlotKpOutput._add_subplot(ax, data=df_niemegk[["kp"]], title=None, width=0.5,
                              ylabel=r'${}$'.format("K_{p}"), align="center",
                              bar_colors=color_niemegk)
    PlotKpOutput._add_subplot(ax, data=df_forecast[["kp"]], title=None, width=0.5,
                              ylabel=r'${}$'.format("K_{p}"), alpha=0.9, bar_colors=color_forecast)

    plt.savefig("./Horizon{}_comparison.png".format(horizon))

    fig2 = plt.figure(figsize=(15, 8))
    ax2 = fig2.add_subplot(1, 1, 1)
    fig2.subplots_adjust(left=0.08, bottom=0.15, right=0.95, top=0.9, wspace=None, hspace=0.6)

    df_total = pd.DataFrame(index=df_niemegk.index)
    df_total["Kp Nowcast"] = df_niemegk["kp"]
    df_total["Kp Forecast"] = df_forecast["kp"]

    ax2 = df_total.plot(drawstyle="steps-post", linewidth=3, ax=ax2, style=['-', '--'],
                        color=['g', 'r'])

    plt.legend(prop={'size': 20})
    # Y-AXIS
    ax2.set_ylim((-0.1, 9.1))
    y_labels = [i for i in range(10) if i % 2 == 0]
    ax2.set_yticks(y_labels)
    ax2.tick_params(axis="y", labelsize=20, direction='in')
    ax2.set_ylabel(r"$K_{p}$", fontsize=20, rotation=90, labelpad=15)
    #first_hour = df_niemegk.index[0].hour

    # def map_dates(x):
    #    if (x.hour - first_hour) % 6 != 0:
    #        return ""
    #    elif ((x.hour - first_hour) % 6 == 0) and (x.hour == first_hour):
    #        return x.strftime("%H:%M\n%d %b")
    #    else:
    #        return x.strftime("%H:%M")

    ax2.set_xlabel("Time (UTC)", fontsize=15, labelpad=10)

    # x_labels = list(df_niemegk.index.map(lambda x: map_dates(x)))

    plt.xticks(fontsize=14)

    # GRID
    ax2.grid(True, axis='y', linestyle='dashed')

    plt.savefig("./Horizon{}_step.png".format(horizon))
