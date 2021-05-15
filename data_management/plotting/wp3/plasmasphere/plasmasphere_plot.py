import os
import subprocess
import glob
import math

import numpy as np
import pandas as pd
from datetime import datetime


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


from data_management.plotting.plotting_base import PlotOutput

basepath = os.path.dirname(__file__)

class PlasmaspherePlot(PlotOutput):

    def __init__(self):

        super().__init__()

        self.figure = None
        self.ax = None
        self.colour_map = mpl.colors.ListedColormap(
            np.load(os.path.abspath(os.path.join(basepath, './my_cmap.npy')))
        )

    @staticmethod
    def _get_date_components(date):
        """
        It gets a datetime instance and returns year, month, day, hour, minute

        :param date: a date
        :type date: an instance of datetime object
        :return: year, month, day, hour, minute
        :rtype: tuple of int
        """
        year = str(date.year)
        month = date.strftime('%m')
        day = date.strftime('%d')
        hour = date.strftime('%H')
        minute = date.strftime("%M")
        return year, month, day, hour, minute

    def _draw_earth_night_side(self):
        self.ax.fill_between(np.linspace(-np.pi / 2, np.pi / 2, 100),
                             0, 1, alpha=1, color='k')

    def _set_axes_ticks_labels(self):

        ticks_loc = [x for x in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
        self.ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        self.ax.set_xticklabels(
            labels=[xlabel for xlabel in np.arange(0, 24, 3)]
        )

        ticks_loc = [1, 2, 3, 4, 5, 6]
        self.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        self.ax.set_ylim(0, 6.5)
        self.ax.set_yticklabels(
            labels=ticks_loc
        )
        self.ax.grid(c='grey', lw=0.5, ls='-', visible=True)

    def _plot_plasma_density(self,
                             angle_values,
                             l_values,
                             density_values
                             ):

        unique_l_values = np.unique(l_values)
        unique_angle_values = np.unique(angle_values)

        angle_grid = np.reshape(angle_values,
                                (len(unique_l_values), len(unique_angle_values)),
                                order="F")
        l_grid = np.reshape(l_values,
                            (len(unique_l_values), len(unique_angle_values)),
                            order="F")
        density_grid = np.reshape(density_values,
                                  (len(unique_l_values), len(unique_angle_values)),
                                  order="F")
        self.ax.contourf(angle_grid, l_grid, density_grid, levels=512, cmap=self.colour_map)

    def _add_colour_bar(self):

        colour_bar_normalization = mpl.colors.Normalize(vmin=0,
                                                        vmax=3.99
                                                        )

        color_bar = plt.colorbar(mpl.cm.ScalarMappable(
            norm=colour_bar_normalization,
            cmap=self.colour_map
        ),
            ax=self.ax,
            use_gridspec=True,
            shrink=0.5,
            pad=0.15
        )
        color_bar.set_label('$log_{10}(n_e)$')

    @staticmethod
    def _mlt_to_angle(mlt):
        return mlt * 2 * math.pi / 24

    def _set_date(self, date):
        if not isinstance(date, datetime):
            raise ValueError("date must be an instance of a datetime object")
        else:
            self.date = date

    def _set_figure(self, fig):
        self.figure = fig

    def _nan_presence(self, density_values):
        if np.sum(np.isnan(density_values)):
            return True

    def _plot_single_plasmasphere(self,  l_values, mlt_values, density_values, date,
              fig_size=(4, 4)):

        self._set_date(date)

        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),
                               figsize=fig_size)
        self.figure = fig
        self.ax = ax

        angle_values = PlasmaspherePlot._mlt_to_angle(mlt_values)

        if not self._nan_presence(density_values):
            self._plot_plasma_density(
                angle_values,
                l_values,
                density_values,
            )

        self._draw_earth_night_side()

        self._set_axes_ticks_labels()

        self.ax.set_title(
            "{}".format(self.date.strftime("%Y-%m-%d, %H:%M")),
        fontdict={'fontweight': 'bold'})

        self._add_colour_bar()

        self.figure.subplots_adjust(left=0.07, right=0.8)
        plt.tight_layout()

    def _save(self, path):
        plt.savefig(path)
        plt.close()

    @staticmethod
    def plot_output(data, output_folder, file_name):
        """
        It produces a video form the output of the plasmasphere predictions.
        In case the predicted densities are nan for some date,
        only the Earth and the basic skeleton appear.

        :param data: instance of pandas DataFrame containing the ouutput of the plasmasphere prediction modules
        :param output_folder: output folder where to store the video, specify as an absolute path
        :param file_name: filename of the video, with extension .mp4
        :return: None
        """

        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas dataframe")

        required_columns = ["L", "MLT", "predicted_densities", "date"]
        for column in required_columns:
            if column not in data.columns:
                raise ValueError("column {} is missing".format(column))

        if not isinstance(data.iloc[0]["date"], datetime):
            raise ValueError("values of date column must be datetime objects")

        if not os.path.isdir(output_folder):
            raise ValueError("specified output_folder doesn't exist")

        if os.path.isfile(os.path.join(output_folder,file_name)):
            os.remove(os.path.join(output_folder, file_name))

        dates = pd.to_datetime(data["date"].unique())
        for date in dates:

            df_date = data[data["date"] == date]

            l_values = df_date["L"].values
            mlt_values = df_date["MLT"].values
            density_values = df_date["predicted_densities"].values
            date = df_date.iloc[0]["date"]

            plotter = PlasmaspherePlot()
            plotter._plot_single_plasmasphere(l_values, mlt_values, density_values, date)

            year, month, day, hour, minute = plotter._get_date_components(date)
            plotter._save(os.path.abspath(os.path.join(output_folder,
                "./plasmasphere_{}_{}_{}_{}_{}.png".format(
                    year, month, day, hour, minute))))

        os.chdir(os.path.abspath(output_folder))
        subprocess.call([
            'ffmpeg', '-framerate', '5', '-i', '%*.png', '-vcodec', 'libx265', '-crf', '28', '-pix_fmt', 'yuv420p',
            file_name
        ])

        for file_name in glob.glob("*.png"):
            os.remove(file_name)

