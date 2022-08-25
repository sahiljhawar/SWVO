import os

import glob
import logging

import subprocess

import math

import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

from data_management.plotting.plotting_base import PlotOutput


class PlasmaspherePlot(PlotOutput):

    def __init__(self, path_c_map=None):

        super().__init__()

        self.figure = None
        self.ax = None
        if path_c_map is None:
            base_path = os.path.dirname(__file__)
            self.path_c_map = os.path.abspath(os.path.join(base_path, 'my_cmap.npy'))
        else:
            self.path_c_map = path_c_map

        self.colour_map = mpl.colors.ListedColormap(np.load(self.path_c_map))

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
            msg = "date must be an instance of a datetime object"
            logging.error(msg)
            raise ValueError(msg)
        else:
            self.date = date

    def _set_figure(self, fig):
        self.figure = fig

    @staticmethod
    def _nan_presence(density_values):
        if np.sum(np.isnan(density_values)):
            return True

    def _plot_single_plasmasphere(self, l_values, mlt_values, density_values, date,
                                  fig_size=(16, 8), df_kp=None, df_solar_wind=None):

        self._set_date(date)
        fig = plt.figure(figsize=fig_size)
        self.figure = fig
        grid = fig.add_gridspec(4, 3)

        ax = self.figure.add_subplot(grid[:, 0], projection='polar')
        self.ax = ax
        angle_values = PlasmaspherePlot._mlt_to_angle(mlt_values)
        if not PlasmaspherePlot._nan_presence(density_values):
            self._plot_plasma_density(
                angle_values,
                l_values,
                density_values,
            )
        self._draw_earth_night_side()
        self._set_axes_ticks_labels()
        self.ax.set_title(
            "{} UTC".format(self.date.strftime("%Y-%m-%d, %H:%M")),
            fontdict={'fontweight': 'bold'})
        self._add_colour_bar()

        if df_kp is not None:
            self.ax_kp = self.figure.add_subplot(grid[0, 1:3])
            self.ax_kp.set_title('Kp')
            locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
            formatter = mdates.ConciseDateFormatter(locator)
            self.ax_kp.xaxis.set_major_locator(locator)
            self.ax_kp.xaxis.set_major_formatter(formatter)
            self.ax_kp.plot(df_kp["t"], df_kp["kp"])
            self.ax_kp.set_ylabel("Kp")
            self.ax_kp.set_xlabel("Time")

        if df_solar_wind is not None:
            self.ax_solar_wind_Bz = self.figure.add_subplot(grid[1, 1:3])
            self.ax_solar_wind_Bz.set_title('Solar wind Bz')
            locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
            formatter = mdates.ConciseDateFormatter(locator)
            self.ax_solar_wind_Bz.xaxis.set_major_locator(locator)
            self.ax_solar_wind_Bz.xaxis.set_major_formatter(formatter)
            self.ax_solar_wind_Bz.plot(df_solar_wind["t"], df_solar_wind["Bz"])
            self.ax_solar_wind_Bz.set_ylabel("Bz")
            self.ax_solar_wind_Bz.set_xlabel("Time")

            self.ax_solar_wind_speed = self.figure.add_subplot(grid[2, 1:3])
            self.ax_solar_wind_speed.set_title('Solar wind speed')
            locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
            formatter = mdates.ConciseDateFormatter(locator)
            self.ax_solar_wind_speed.xaxis.set_major_locator(locator)
            self.ax_solar_wind_speed.xaxis.set_major_formatter(formatter)
            self.ax_solar_wind_speed.plot(df_solar_wind["t"], df_solar_wind["speed"])
            self.ax_solar_wind_speed.set_ylabel("Speed")
            self.ax_solar_wind_speed.set_xlabel("Time")

            self.ax_solar_wind_proton_density = self.figure.add_subplot(grid[3, 1:3])
            self.ax_solar_wind_proton_density.set_title("Solar wind proton density")
            locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
            formatter = mdates.ConciseDateFormatter(locator)
            self.ax_solar_wind_proton_density.xaxis.set_major_locator(locator)
            self.ax_solar_wind_proton_density.xaxis.set_major_formatter(formatter)
            self.ax_solar_wind_proton_density.plot(df_solar_wind["t"], df_solar_wind["proton_density"])
            self.ax_solar_wind_proton_density.set_xlabel("Time")
            self.ax_solar_wind_proton_density.set_ylabel("Proton density")

        plt.tight_layout()

    @staticmethod
    def _save(path):
        plt.savefig(path)
        plt.close()

    @staticmethod
    def plot_output(data, output_folder, video_file_name, kp_inputs, solar_wind_inputs):
        """
        It produces a video form the output of the plasmasphere predictions.
        In case the predicted densities are nan for some date,
        only the Earth and the basic skeleton appear.

        :param data: instance of pandas DataFrame containing the output of the plasmasphere prediction modules
        :type data: pandas.DataFrame
        :param output_folder: output folder where to store the video, specify as an absolute path
        :type output_folder: str
        :param video_file_name: filename of the video, with extension .mp4
        :type video_file_name: str
        """

        if not isinstance(data, pd.DataFrame):
            msg = "data must be an instance of a pandas dataframe, " \
                  "instead it is of type {}".format(type(data))
            logging.error(msg)
            raise TypeError(msg)

        required_columns = ["L", "MLT", "predicted_densities", "t"]
        for column in required_columns:
            if column not in data.columns:
                msg = "column {} is missing".format(column)
                logging.error(msg)
                raise ValueError(msg)

        if not isinstance(data.iloc[0]["t"], datetime):
            msg = "values of date column must be datetime objects"
            logging.error(msg)
            raise TypeError(msg)

        if not os.path.isdir(output_folder):
            msg = "specified output_folder doesn't exist"
            logging.error(msg)
            raise FileNotFoundError(msg)

        output_folder = os.path.abspath(output_folder)

        if os.path.isfile(os.path.join(output_folder, video_file_name)):
            os.remove(os.path.join(output_folder, video_file_name))

        temp_folder_path = os.path.join(output_folder, "temp_with_inputs/")
        if os.path.exists(temp_folder_path):
            temp_folder_files = glob.glob(os.path.join(temp_folder_path, "*.png"))
            for file in temp_folder_files:
                os.remove(file)
        else:
            os.makedirs(temp_folder_path)

        logging.info("Started individual plasmasphere reconstructions generation")
        dates = pd.to_datetime(data["t"].unique())
        for date in dates:
            df_date = data[data["t"] == date]

            l_values = df_date["L"].values
            mlt_values = df_date["MLT"].values
            density_values = df_date["predicted_densities"].values
            date = df_date.iloc[0]["t"]
            if not PlasmaspherePlot._nan_presence(density_values):
                df_kp_date = None
                solar_wind_date = None
                if kp_inputs is not None and solar_wind_inputs is not None:
                    df_kp_date = kp_inputs[kp_inputs["date_of_run"] == date]
                    solar_wind_date = solar_wind_inputs[solar_wind_inputs["date_of_run"] == date]

                plotter = PlasmaspherePlot()
                plotter._plot_single_plasmasphere(l_values, mlt_values,
                                                  density_values, date,
                                                  df_kp=df_kp_date,
                                                  df_solar_wind=solar_wind_date)

                year, month, day, hour, minute = plotter._get_date_components(date)
                plotter._save(os.path.join(temp_folder_path,
                                           "./plasmasphere_{}_{}_{}_{}_{}.png".format(
                                               year, month, day, hour, minute)
                                           )
                              )
                logging.info("for date {} plot has been generated".format(date))
            else:
                logging.warning("for date {} the predictions were NaN and "
                                "the plot has not been generated".format(date))
        logging.info("Finished individual plasmasphere reconstructions generation")

        logging.info("Starting video generation")

        if not len(os.listdir(temp_folder_path)) == 0:
            os.chdir(temp_folder_path)
            subprocess.check_call([
                'ffmpeg', '-framerate', '3', '-i', os.path.join(temp_folder_path, "%*.png"), '-vcodec', 'libx264', '-crf',
                '28', '-pix_fmt', 'yuv420p',
                os.path.join(output_folder, video_file_name)
            ])
            logging.info("Finished video generation and saving")

            for file_name in glob.glob(os.path.join(temp_folder_path, "*.png")):
                os.remove(file_name)
        else:
            logging.info("All predictions were Nan and no movie has been generated")
