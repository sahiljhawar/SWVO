import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import pandas as pd
import numpy as np
import logging
from data_management.plotting.plotting_base import PlotOutput

matplotlib.use('Agg')


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
                     ylabel=r"$K_{p}$", bar_colors=None, data_column="kp"):

        if bar_colors is None:
            bar_colors = PlotKpOutput._add_bar_color(data, list(data.keys())[0])
        ax = data[data_column].plot(kind="bar", ax=ax, edgecolor=['k'] * len(data), color=bar_colors,
                                    align=align, width=width, legend=False, alpha=alpha)
        ax.set_title(title, fontsize=title_font)

        ax.set_ylim(ylim)
        y_labels = [i for i in range(10) if i % 2 == 0]
        ax.set_yticks(y_labels)
        ax.tick_params(axis="y", labelsize=ylabel_fontsize, direction='in')
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, rotation=90, labelpad=15)
        first_hour = data.index[0].hour

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
            data_column = "kp"
        else:
            label = "H_{p}60"
            data_column = "Hp60"

        if ax is None:
            fig = plt.figure(figsize=(15, 8))
            ax = fig.add_subplot(1, 1, 1)

        ax = PlotKpOutput._add_subplot(ax, data=data[[data_column]],
                                       title=None,
                                       ylabel=r'${}$'.format(label),
                                       data_column=data_column)

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
    def _add_max_bars(ax, data, color='r'):
        for i, d in enumerate(data.index):
            ax.hlines(y=data.values[i][0], xmin=i + 0.03, xmax=i + 1 - 0.1, linewidth=4, color=color)
        return ax

    @staticmethod
    def plot_output(data, legend=True):
        """
        This function plots output data for Kp Ensemble. The plot format is at the moment
        fixed.

        :param data: This is the standard output format of Kp Ensemble products read by KpEnsembleReader class.
        :type data: list of pandas.DataFrame
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

        label = "K_p"

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)

        plotter = PlotKpOutput()
        ax = plotter.plot_output(data_median, ax=ax)
        ax = PlotKpEnsembleOutput._add_max_bars(ax, data=data_max)

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
