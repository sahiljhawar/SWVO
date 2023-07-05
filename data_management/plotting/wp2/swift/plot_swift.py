import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from data_management.plotting.plotting_base import PlotOutput

matplotlib.use('Agg')


class PlotSWIFTOutput(PlotOutput):
    """
    This class is in charge to produce standard plots for SWIFT WP2 output data.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _add_subplot(ax, data, title=None, title_font=9, ylabel_fontsize=10, ylabel=None, xticks_labels_show=False,
                     line_width=3, color='orange', legend=False, label="SWIFT"):

        ax = data.plot(ax=ax, legend=legend, linewidth=line_width, color=color, label=label)
        ax.set_title(title, fontsize=title_font)

        # Y-AXIS
        ax.tick_params(axis="y", labelsize=ylabel_fontsize, direction='in')
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, rotation=90, labelpad=0)

        # X-AXIS
        ax.tick_params(axis="x", which="both", bottom=True)

        # GRID
        ax.grid(True, axis='y', linestyle='dashed')

        return ax

    @staticmethod
    def plot_output(data, color="orange", legend=False, label="SWIFT"):
        """
        This function plots output data for SWIFT solar wind variables. The plot format is at the moment
        fixed.

        :param data: This is the standard output format of SWIFT products read by SWIFTReader class.
        :type data: pandas.DataFrame
        :return: A figure object of type matplotlib.figure.Figure containing the produced plot
        """
        density = data["proton_density"]
        speed = data["speed"]
        b = data["b"]
        temperature = data["temperature"]

        fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10, 10))

        fig.supxlabel("Time (UTC)", fontsize=15)

        locator = matplotlib.dates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = matplotlib.dates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        def map_dates(x):
            if x.hour == 0:
                return x.strftime("%Y-%m-%d")
            else:
                return ""
        # ax.set_xticks([])
        # ax.set_xticks(data.index[::12])
        # x_labels = list(data.index[::12].map(lambda x: map_dates(x)))
        # ax.set_xticklabels(labels=x_labels, rotation=30, fontsize=10)

        # ax.set_xlabel("Time (UTC)", fontsize=15, labelpad=0)
        # ax.set_xlabel("")

        PlotSWIFTOutput._add_subplot(ax[0], speed, ylabel=r"$|U|(km/s)$",
                                     color=color, legend=legend,
                                     label="SWIFT")
        PlotSWIFTOutput._add_subplot(ax[1], density,
                                     ylabel=r"$N_{p}(cm^{-3})$",
                                     color=color, legend=legend,
                                     label="SWIFT")
        PlotSWIFTOutput._add_subplot(ax[2], temperature,
                                     ylabel=r"$Temperature(K)$",
                                     color=color, legend=legend,
                                     label="SWIFT")
        PlotSWIFTOutput._add_subplot(ax[3], b, ylabel=r"$|B|(nT)$",
                                     xticks_labels_show=True, color=color,
                                     legend=legend, label="SWIFT")

        plt.tight_layout()

        return fig

    @staticmethod
    def plot_ensemble_output(data):
        """
        This function plots output data for SWIFT ensemble solar wind variables.

        :param data: This is the standard output format of SWIFT Ensemble products read by SWIFTEnsembleReader class.
        :type data: list of pandas.DataFrame
        :return: A figure object of type matplotlib.figure.Figure containing the produced plot
        """
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(4, 1)
        gs.update(left=0.1, right=0.9, wspace=0.05, hspace=0.1)
        axis = {}
        for i in range(4):
            axis["ax{}".format(i + 1)] = plt.subplot(gs[i, 0])

        for d in data:
            density = d["proton_density"]
            speed = d["speed"]
            b = d["b"]
            temperature = d["temperature"]

            PlotSWIFTOutput._add_subplot(axis["ax1"], density, ylabel=r"$N_{p}(cm^{-3})$", line_width=1, color="b")
            PlotSWIFTOutput._add_subplot(axis["ax2"], speed, ylabel=r"$|U|(km/s)$", line_width=1)
            PlotSWIFTOutput._add_subplot(axis["ax3"], temperature, ylabel=r"$Temperature(K)$", line_width=1)
            PlotSWIFTOutput._add_subplot(axis["ax4"], b, ylabel=r"$|B|(nT)$", xticks_labels_show=True, line_width=1)

        plt.tight_layout()
        return fig
