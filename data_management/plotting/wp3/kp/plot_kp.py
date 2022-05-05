import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import pandas as pd
import numpy as np
import logging
from data_management.plotting.plotting_base import PlotOutput


# matplotlib.use('Agg')


class PlotKpOutput(PlotOutput):
    """
    This class is in charge to produce standard plots for Kp and geomagnetic indexes output data.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _add_bar_color(data, key):
        color = []
        for i in range(len(data)):
            if data[key][i] < 4:
                color.append([0.0, 0.5, 0.0, 1.0])
            elif data[key][i] == 4:
                color.append([204 / 255.0, 204 / 255.0, 0.0, 1.0])
            elif data[key][i] > 4:
                color.append([1.0, 0.0, 0.0, 1.0])
            elif data[key][i] >= 9.5:
                color.append([0.0, 0.0, 0.0, 0.1])
        return color

    @staticmethod
    def _add_subplot(ax, data, title=None, rotation=0, title_font=9, xlabel_fontsize=14,
                     ylabel_fontsize=20, ylim=(-0.1, 9.1), width=0.9, align="edge", alpha=None,
                     ylabel=r"$K_{p}$", bar_colors=None):
        # PLOT
        if bar_colors is None:
            bar_colors = PlotKpOutput._add_bar_color(data, list(data.keys())[0])
        ax = data["kp"].plot(kind="bar", ax=ax, edgecolor=['k'] * len(data), color=bar_colors,
                             align=align, width=width, legend=False, alpha=alpha)
        # TITLE
        ax.set_title(title, fontsize=title_font)

        # Y-AXIS
        ax.set_ylim(ylim)
        y_labels = [i for i in range(10) if i % 2 == 0]
        ax.set_yticks(y_labels)
        ax.tick_params(axis="y", labelsize=ylabel_fontsize, direction='in')
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, rotation=90, labelpad=15)
        first_hour = data.index[0].hour

        # X-AXIS
        def map_dates(x):

            if (x.hour - first_hour) % 6 != 0:
                return ""
            elif ((x.hour - first_hour) % 6 == 0) and (x.hour == first_hour):
                return x.strftime("%H:%M\n%d %b")
            else:
                return x.strftime("%H:%M")

        ax.set_xlabel("Time (UTC)", fontsize=15, labelpad=10)

        x_labels = list(data.index.map(lambda x: map_dates(x)))
        ax.set_xticklabels(labels=x_labels, rotation=rotation, fontsize=xlabel_fontsize)

        # GRID
        ax.grid(True, axis='y', linestyle='dashed')

        return ax

    @staticmethod
    def plot_output(data, ax=None, legend=True):
        """
        This function plots output data for Kp products. The plot format is at the moment fixed.

        :param data: This is the standard output format of Kp products read by KpReader class.
        :type data: pandas.DataFrame
        :param ax: An Axes object in the case the plot needs to be combined with other figures outside of the class
                   otherwise pass None.
        :type ax: matplotlib.axes.Axes or None
        :param legend: If True the default legend of the plot is kept, otherwise it is not plotted.
        :type legend: bool
        :return: An Axes object
        """

        if not isinstance(data, pd.DataFrame):
            msg = "data must be an instance of a pandas dataframe, instead it is of type {}".format(type(data))
            logging.error(msg)
            raise TypeError(msg)

        # Todo This is a hack, still working on it
        cadence = (data.index[1] - data.index[0]).seconds // 3600
        if cadence == 3:
            label = "K_{p}"
        else:
            label = "H_{p}"

        if ax is None:
            fig = plt.figure(figsize=(15, 8))
            ax = fig.add_subplot(1, 1, 1)
        ax = PlotKpOutput._add_subplot(ax, data=data[["kp"]], title=None, ylabel=r'${}$'.format(label))

        red_patch = patches.Patch(color='red', label=r'${}$ > 4'.format(label))
        yellow_patch = patches.Patch(color=[204 / 255.0, 204 / 255.0, 0.0, 1.0], label=r'${}$ = 4'.format(label))
        green_patch = patches.Patch(color='green', label=r'${}$ < 4'.format(label))
        transparent_patch = patches.Patch(color=[0, 0, 0, 0.1], label='Data not available')

        if legend:
            ax.legend(bbox_to_anchor=(0., 1., 0.84, .275),
                      handles=[green_patch, yellow_patch, red_patch, transparent_patch],
                      ncol=4, fontsize="x-large", shadow=True)

        if ax is None:
            fig.subplots_adjust(left=None, bottom=0.3, right=None, top=0.7, wspace=None, hspace=0.6)

        return ax


class PlotKpEnsembleOutput(PlotOutput):
    """
    This class is in charge to produce standard plots for Kp and geomagnetic indexes output data.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _add_plot_max(ax, data, color='r', style="-"):

        x = data.index
        y = data.values

        y_slice = y[:-1]
        X = np.c_[x[:-1], x[1:], x[1:]]
        Y = np.c_[y_slice, y_slice, np.zeros_like(y_slice) * np.nan]

        data_new = pd.DataFrame(Y.flatten(), index=X.flatten(), columns=["kp"])
        # TODO Not working well still
        ax = data_new['kp'].plot(drawstyle="steps-post", linewidth=3, ax=ax, style=[style], color=[color])
        # ax = data['kp'].plot(kind="hlines", linewidth=3, ax=ax, style=[style], color=[color])

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        return ax

    @staticmethod
    def plot_output(data, legend=True):
        """
        This function plots output data for Kp Ensemble. The plot format is at the moment
        fixed.

        :param data: This is the standard output format of Kp Ensemble products read by KpEnsembleReader class.
        :type data: list of pandas.DataFrame
        :param ax: An Axes object in the case the plot needs to be combined with other figures outside of the class
                   otherwise pass None.
        :type ax: matplotlib.axes.Axes or None
        :param legend: If True the default legend of the plot is kept, otherwise it is not plotted.
        :type legend: bool
        :return: An Axes object
        """

        if not isinstance(data, list):
            msg = "Data argument passed must be an instance of a list, instead it is of type {}".format(type(data))
            logging.error(msg)
            raise TypeError(msg)

        for d in data:
            if not isinstance(d, pd.DataFrame):
                msg = "Each element of data must be an instance of a pandas dataframe, instead" \
                      " it is of type {}".format(type(d))
                logging.error(msg)
                raise TypeError(msg)

        data_median = pd.DataFrame(np.median([d.values.flatten() for d in data], axis=0), columns=["kp"],
                                   index=data[0].index)
        data_max = pd.DataFrame(np.max([d.values.flatten() for d in data], axis=0), columns=["kp"], index=data[0].index)
        data_max.xs(data_max.index[3])["kp"] = 4
        data_min = pd.DataFrame(np.min([d.values.flatten() for d in data], axis=0), columns=["kp"], index=data[0].index)

        label = "K_p"

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax2 = ax.twiny()

        ax = PlotKpOutput._add_subplot(ax, data=data_median, title=None, ylabel=r'${}$'.format(label))
        ax2 = PlotKpEnsembleOutput._add_plot_max(ax2, data=data_max)

        red_patch = patches.Patch(color='red', label=r'${}$ > 4'.format(label))
        yellow_patch = patches.Patch(color=[204 / 255.0, 204 / 255.0, 0.0, 1.0], label=r'${}$ = 4'.format(label))
        green_patch = patches.Patch(color='green', label=r'${}$ < 4'.format(label))
        transparent_patch = patches.Patch(color=[0, 0, 0, 0.1], label='Data not available')

        if legend:
            ax.legend(bbox_to_anchor=(0., 1., 0.84, .275),
                      handles=[green_patch, yellow_patch, red_patch, transparent_patch],
                      ncol=4, fontsize="x-large", shadow=True)

        if ax is None:
            fig.subplots_adjust(left=None, bottom=0.3, right=None, top=0.7, wspace=None, hspace=0.6)

        return ax
