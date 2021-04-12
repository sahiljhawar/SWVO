import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from data_management.plotting.plotting_base import PlotOutput

matplotlib.use('Agg')


class PlotKpOutput(PlotOutput):
    """
    This class is in charge to produce standard plots for Kp and geomagnetic indexes output data.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def add_bar_color(data, key):
        color = []
        for i in range(len(data)):
            if data[key][i] < 4:
                color.append('g')
            elif data[key][i] == 4:
                color.append([204 / 255.0, 204 / 255.0, 0.0, 1.0])
            elif data[key][i] > 4:
                color.append('r')
            elif data[key][i] >= 9.5:
                color.append([0.0, 0.0, 0.0, 0.1])
        return color

    @staticmethod
    def add_subplot(ax, data, title=None, rotation=0, title_font=9, xlabel_fontsize=14,
                    ylabel_fontsize=20, ylim=(-0.1, 9.1),
                    ylabel=r"$K_{p}$", cadence=3):
        # PLOT
        bar_colors = PlotKpOutput.add_bar_color(data, list(data.keys())[0])
        ax = data.plot(kind="bar", ax=ax, edgecolor=['k'] * len(data), color=[bar_colors],
                       align="edge", width=0.9, legend=False)
        # TITLE
        ax.set_title(title, fontsize=title_font)

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

        return ax

    @staticmethod
    def plot_output(data) -> matplotlib.figure.Figure:
        """
        This function plots output data for Kp and other geomagnetic index products. The plot format is at the moment
        fixed.

        :param data: This is the standard output format of Kp products read by KpReader class.
        :type data: pandas.DataFrame
        :return: A figure object of type matplotlib.figure.Figure containing the produced plot
        """
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax = PlotKpOutput.add_subplot(ax, data=data[["kp"]], title=None,
                                      cadence=(data.index[1] - data.index[0]).seconds // 3600)

        red_patch = patches.Patch(color='red', label=r'$K_{p}$ > 4')
        yellow_patch = patches.Patch(color=[204 / 255.0, 204 / 255.0, 0.0, 1.0], label=r'$K_{p}$ = 4')
        green_patch = patches.Patch(color='green', label=r'$K_{p}$ < 4')
        transparent_patch = patches.Patch(color=[0, 0, 0, 0.1], label='Data not available')
        ax.legend(bbox_to_anchor=(0., 1., 0.84, .275),
                  handles=[green_patch, yellow_patch, red_patch, transparent_patch],
                  ncol=4, fontsize="x-large", shadow=True)

        fig.subplots_adjust(left=None, bottom=0.3, right=None, top=0.7, wspace=None, hspace=0.6)
        return fig


if __name__ == "__main__":
    from data_management.io.wp3.read_kp import KPReader

    reader = KPReader()
    data, _ = reader.read(source="swpc")
    fig = PlotKpOutput.plot_output(data)
    print (type(fig))

    figure = matplotlib.figure.Figure()
