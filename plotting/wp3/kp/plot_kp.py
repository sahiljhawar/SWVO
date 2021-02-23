import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib

matplotlib.use('Agg')

import os
import sys
import argparse
import datetime as dt
import logging

sys.path.append("/PAGER/WP8/data_management/io/wp3/")
from read_kp import KPReader

DATA_PATH = "/PAGER/WP3/data/outputs/"
#RESULTS_PATH = "/PAGER/WP3/data/figures/"
RESULTS_PATH = "/home/ruggero/temp/"

def add_bar_color(data, key):
    color = []
    for i in range(len(data)):
        if data[key][i] < 4:
            color.append('g')
        elif data[key][i] == 4:
            color.append([204 / 255.0, 204 / 255.0, 0.0, 1.0])
            # color.append('y')
        elif data[key][i] > 4:
            color.append('r')
        # elif data[key][i] >= 9.5:
        #    color.append([0.0, 0.0, 0.0, 0.1])
    return color


def add_subplot(ax, data, title=None, rotation=0, title_font=9, xlabel_fontsize=14,
                ylabel_fontsize=20, ylim=(-0.1, 9.1), ylabel=r"$K_{p}$", cadence=3):
    # PLOT
    bar_colors = add_bar_color(data, list(data.keys())[0])
    data.plot(kind="bar", ax=ax, edgecolor=['k'] * len(data), color=[bar_colors],
              align="edge", width=0.9, legend=False)
    # TITLE
    plt.title(title, fontsize=title_font)

    # Y-AXIS
    ax.set_ylim(ylim)
    y_labels = [i for i in range(10) if i % 2 == 0]
    ax.set_yticks(y_labels)
    ax.tick_params(axis="y", labelsize=ylabel_fontsize, direction='in')
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, rotation=90, labelpad=15)

    n_points = len(data)
    if n_points > 12:
        cadence *= 2

    # X-AXIS
    def map_dates(x):
        if x.hour % cadence != 0:
            return ""
        if (x.hour % cadence == 0) and (x.minute == 0):
            return x.strftime("%H:%M\n%d %b")
        else:
            return x.strftime("%H:%M")

    ax.set_xlabel("Time (UTC)", fontsize=15, labelpad=10)

    x_labels = list(data.index.map(lambda x: map_dates(x)))
    ax.set_xticklabels(labels=x_labels, rotation=rotation, fontsize=xlabel_fontsize)

    # GRID
    ax.grid(True, axis='y', linestyle='dashed')


def plot_forecast(model_data):
    _ = plt.figure(figsize=(15, 8))
    ax = plt.subplot(1, 1, 1)
    add_subplot(ax, data=model_data[["kp"]], title=None,
                cadence=(model_data.index[1] - model_data.index[0]).seconds // 3600)

    red_patch = mpatches.Patch(color='red', label=r'$K_{p}$ > 4')
    yellow_patch = mpatches.Patch(color=[204 / 255.0, 204 / 255.0, 0.0, 1.0], label=r'$K_{p}$ = 4')
    green_patch = mpatches.Patch(color='green', label=r'$K_{p}$ < 4')
    transparent_patch = mpatches.Patch(color=[0, 0, 0, 0.1], label='Data not available')
    plt.legend(bbox_to_anchor=(0., 1., 0.84, .275), handles=[green_patch, yellow_patch, red_patch, transparent_patch],
               ncol=4, fontsize="x-large", shadow=True)

    plt.subplots_adjust(left=None, bottom=0.3, right=None, top=0.7,
                        wspace=None, hspace=0.6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-date', action="store", default=None, type=str,
                        help="Requested date to plot in the format %YYYY-%mm-%dd")
    args = parser.parse_args()
    plotting_date = dt.datetime.strptime(args.date, "%Y-%m-%d")

    reader = KPReader()

    try:
        data_niemegk, date = reader.read("niemegk", plotting_date)
        plot_forecast(data_niemegk)
        plt.savefig(os.path.join(RESULTS_PATH, "Niemegk_LAST.png"))
        plt.savefig(os.path.join(RESULTS_PATH, "Niemegk_{}.png".format(date.strftime("%Y%m%d"))))
    except TypeError:
        logging.error("Data for Niemegk nowcast for date {} not found...impossible to produce data plot...".format(plotting_date))

    try:
        data_swpc, date = reader.read("swpc", plotting_date)
        plot_forecast(data_swpc)
        plt.savefig(os.path.join(RESULTS_PATH, "SWPC_LAST.png"))
        plt.savefig(os.path.join(RESULTS_PATH, "SWPC_{}.png".format(date.strftime("%Y%m%d"))))
    except TypeError:
        logging.error("Data for SWPC forecast for date {} not found...impossible to produce data plot...".format(plotting_date))


#    data_l1, date = reader.read("l1", plotting_date)
#    plot_forecast(data_l1)
#    plt.savefig(os.path.join(RESULTS_PATH, "L1_LAST.png"))
#    plt.savefig(os.path.join(RESULTS_PATH, "L1_{}.png".format(date.strftime("%Y%m%d"))))

#    data_swift, date = reader.read("swift", plotting_date)
#    plot_forecast(data_swift)
#    plt.savefig(os.path.join(RESULTS_PATH, "SWIFTKP_LAST.png"))
#    plt.savefig(os.path.join(RESULTS_PATH, "SWIFTKP_{}.png".format(date.strftime("%Y%m%d"))))
