import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from data_management.plotting.plotting_base import PlotOutput


# matplotlib.use('Agg')


class PlotSWIFTOutput(PlotOutput):
    """
    This class is in charge to produce standard plots for SWIFT WP2 output data.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _add_subplot(ax, data, title=None, title_font=9, xlabel_fontsize=10,
                     ylabel_fontsize=10, ylabel=None, xticks_labels_show=False):

        ax = data.plot(ax=ax, legend=False, linewidth=3)
        ax.set_title(title, fontsize=title_font)

        # Y-AXIS
        ax.tick_params(axis="y", labelsize=ylabel_fontsize, direction='in')
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, rotation=90, labelpad=0)

        # X-AXIS
        def map_dates(x):
            if x.hour == 0:
                return x.strftime("%Y-%m-%d")
            else:
                return ""

        ax.set_xlabel("Time (UTC)", fontsize=15, labelpad=0)

        if xticks_labels_show:
            ax.set_xticks(data.index[::12])
            x_labels = list(data.index[::12].map(lambda x: map_dates(x)))
            ax.set_xticklabels(labels=x_labels, rotation=30, fontsize=xlabel_fontsize)
            ax.set_xlabel("Time (UTC)", fontsize=15, labelpad=0)
        else:
            ax.tick_params(axis="x", which="both", bottom=False)
            ax.set_xticklabels(labels=[])
            ax.set_xlabel("")

        # GRID
        ax.grid(True, axis='y', linestyle='dashed')

        return ax

    @staticmethod
    def plot_output(data):
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

        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(4, 1)
        gs.update(left=0.1, right=0.9, wspace=0.05, hspace=0.1)
        axis = {}
        for i in range(4):
            axis["ax{}".format(i + 1)] = plt.subplot(gs[i, 0])

        PlotSWIFTOutput._add_subplot(axis["ax1"], density, ylabel=r"$N_{p}(cm^{-3})$")
        PlotSWIFTOutput._add_subplot(axis["ax2"], speed, ylabel=r"$|U|(km/s)$")
        PlotSWIFTOutput._add_subplot(axis["ax3"], temperature, ylabel=r"$Temperature(K)$")
        PlotSWIFTOutput._add_subplot(axis["ax4"], b, ylabel=r"$|B|(nT)$", xticks_labels_show=True)

        return fig


if __name__ == "__main__":
    from data_management.io.wp2.read_swift import SwiftReader
    import datetime as dt

    reader = SwiftReader()
    data, _ = reader.read(dt.datetime(2021, 6, 1))
    plotter = PlotSWIFTOutput()
    fig = plotter.plot_output(data)
    plt.show()
