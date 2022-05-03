import os
import argparse
import datetime as dt
import matplotlib.pyplot as plt
import logging
import pandas as pd
import matplotlib.patches as patches

from data_management.io.wp3.read_kp import KPReader
from data_management.plotting.wp3.kp.plot_kp import PlotKpOutput


def read_niemegk_data(s_date, e_date, input_folder):
    reader = KPReader(input_folder)
    index = 0
    data = []
    while True:
        data_niemegk, _ = reader.read("niemegk", s_date + dt.timedelta(days=index), header=False)
        if data_niemegk is not None:
            if index > 0:
                data_niemegk = data_niemegk[data_niemegk.index > max(data[-1].index)]
            data.append(data_niemegk)
            if max(data_niemegk.index) > e_date:
                break
        index += 1
    data = pd.concat(data, axis=0)
    data = data[data.index <= e_date]
    data = data[data.index >= s_date]
    return data


def read_forecast_data(s_date, e_date, horizon, source, model_name, input_folder):
    reader = KPReader(input_folder)
    values = []
    dates = []
    index = 0
    while True:
        data_forecast, _ = reader.read(source, s_date + dt.timedelta(hours=3 * index - horizon), model_name=model_name,
                                       header=False)
        forecast_value = data_forecast[data_forecast.index == s_date +
                                       dt.timedelta(hours=3 * index)]["kp"].values[0]
        values.append(forecast_value)
        dates.append(s_date + dt.timedelta(hours=(3 * index)))
        index += 1
        if s_date + dt.timedelta(hours=3 * index) > e_date:
            break
    return pd.DataFrame({"kp": values}, index=dates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-start_date', action="store", default=None, type=str,
                        help="Requested date to plot in the format %YYYY%mm%dd%HH")
    parser.add_argument('-end_date', action="store", default=None, type=str,
                        help="Requested date to plot in the format %YYYY%mm%dd%HH")
    parser.add_argument('-output', action="store", default="/PAGER/WP3/data/figures/comparison/", type=str,
                        help="Path to a folder where to store the produced figures")
    parser.add_argument('-input', action="store", default="/PAGER/WP3/data/outputs/", type=str,
                        help="Path to a folder where the data to plot is stored...(be more precise)")
    parser.add_argument('-model', action="store", default="KP-FULL-SW-PAGER", type=str,
                        help="Kp model name")
    parser.add_argument('-horizon', action="store", default=None, type=int,
                        help="Forecast horizon for Kp as integer multiple of 3 starting from 0")
    # parser.add_argument('-log', action="store", default=None, type=str,
    #                    help="Log directory if logging is to be enabled.")

    args = parser.parse_args()

    date_now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    if (args.start_date is None) or (args.end_date is None):
        start_date = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0) - dt.timedelta(hours=24)
        end_date = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        logging.warning("One or more dates not provide. Using default current date for an interval of one day")
    else:
        try:
            start_date = dt.datetime.strptime(args.start_date, "%Y%m%d%H")
            end_date = dt.datetime.strptime(args.end_date, "%Y%m%d%H")
            assert args.end_date > args.start_date
        except ValueError:
            msg = "At least one of the provided dates {} not in correct format %Y%m%d%H. Aborting...".format(args.date)
            logging.error(msg)
            raise RuntimeError(msg)
        except AssertionError:
            msg = "End date {} is smaller of start date {}. Aborting...".format(args.end_date, args.start_date)
            logging.error(msg)
            raise AssertionError(msg)

    # if args.logdir is not None:
    #    log_file = "wp3_plot_all_kp_{}.log".format(plotting_date.strftime("%Y%m%dT%H%M%S"))
    #    logging.basicConfig(filename=os.path.join(args.logdir, log_file), level=logging.INFO,
    #                        datefmt="%Y-%m-%d %H:%M:%S",
    #                        format="%(asctime)s;%(levelname)s;%(message)s")

    RESULTS_PATH = args.output
    DATA_PATH = args.input

    df_niemegk = read_niemegk_data(start_date, end_date, DATA_PATH)
    df_forecast = read_forecast_data(start_date, end_date, source="l1", horizon=args.horizon, model_name=args.model,
                                     input_folder=DATA_PATH)

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

    plt.savefig(os.path.join(args.output, "Comparison_Bar_Kpmodel_{}_hor_{}.png".format(args.model, args.horizon)))

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
    ax2.set_xlabel("Time (UTC)", fontsize=15, labelpad=10)
    plt.xticks(fontsize=14)
    ax2.grid(True, axis='y', linestyle='dashed')

    plt.savefig(os.path.join(args.output, "Comparison_Step_Kpmodel_{}_hor_{}.png".format(args.model, args.horizon)))
