import sys
from os import path

import math
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from datetime import datetime


basepath = path.dirname(__file__)
sys.path.append(path.abspath(path.join(basepath, "..")))
from plotting.plotting_base import BasePlot


class PlasmaspherePlot(BasePlot):

    def _draw_earth_night_side(self):
        """
        It draws the night side of the earth

        :param ax:
        :return:
        """

        Ls = np.linspace(0, 1, 100)
        angles = np.linspace(-math.pi / 2, math.pi / 2, 600)

        angle_matrix, L_matrix = np.meshgrid(angles, Ls)

        angle_column = angle_matrix.flatten('F').reshape(-1, 1)
        L_column = L_matrix.flatten('F').reshape(-1, 1)

        self._get_ax().scatter(angle_column, L_column, s=0.1, c="black")

    def _draw_earth_day_side(self):
        """
        It draws the day side of the earth

        :param ax:
        :return:
        """

        Ls = np.linspace(0, 1, 20)
        angles = np.linspace(math.pi / 2, 3 * math.pi / 2, 150)

        angle_matrix, L_matrix = np.meshgrid(angles, Ls)

        angle_column = angle_matrix.flatten('F').reshape(-1, 1)
        L_column = L_matrix.flatten('F').reshape(-1, 1)

        self._get_ax().scatter(angle_column, L_column, s=0.1, c="w")

    def _set_style(self):

        # set stile
        self._get_ax().set_ylim(0, 6)
        self._get_ax().set_xticklabels(
            [xlabel for xlabel in np.arange(0, 24, 3)],
            fontdict={'fontweight': "roman", 'fontsize': 18}
        )
        self._get_ax().set_yticklabels(
            [1, 2, 3, 4, 5, 6],
            fontdict={'fontweight': "roman", 'fontsize': 16})
        self._get_ax().grid(c='w', lw=1.5, ls='-', visible=True)

    def _plot_plasma_density(self,
                             angle_values,
                             l_values,
                             density_values):
        self._get_ax().scatter(angle_values,
                                   l_values,
                                   c=density_values,
                                   s=l_values,
                                   cmap=self._get_colour_map(),
                                   alpha=0.4
                                   )

        #vmin, vmax = c.get_clim()

        #return vmin, vmax

    def _set_title_coordinates(self, title_coordinates):
        self.title_coordinates = title_coordinates

    def _get_title_coordinates(self):
        return self.title_coordinates

    def _set_colour_bar_axes(self, axes):
        self.colour_bar_axes = axes

    def _get_colour_bar_axes(self):
        return self.colour_bar_axes

    def _add_colour_bar(self):

        self._set_colour_bar_axes(self._get_figure().add_axes([0.9, 0.2, 0.04, 0.6]))

        colour_bar_normalization = mpl.colors.Normalize(vmin=0,
                                                        vmax=3.9
                                                        )
        cb = mpl.colorbar.ColorbarBase(self._get_colour_bar_axes(),
                                       cmap=self._get_colour_map(),
                                       norm=colour_bar_normalization)
        cb.set_label('log10(n\u2091), 1/cm\u00b3 ')

        self._get_colour_bar_axes().tick_params(direction="in")
        self._get_colour_bar_axes().tick_params(axis='both', which='major', labelsize=18)

    def _set_colour_map(self):
        self.colour_map = mpl.colors.ListedColormap(np.load('./my_cmap.npy'))

    def _get_colour_map(self):
        return self.colour_map

    @staticmethod
    def _mlt_to_angle(mlt):
        return mlt * 2 * math.pi / 24

    def _set_date(self, date):
        if not isinstance(date, datetime):
            raise ValueError("date must be an instance of a datetime object")
        else:
            self.date = date

    def _get_date(self):
        return self.date

    def _set_figure(self, fig):
        self.figure = fig

    def _get_figure(self):
        return self.figure

    def _set_ax(self, ax):
        self.ax = ax

    def _get_ax(self):
        return self.ax

    def _check_inputs(self, density_values):
        if np.sum(np.isnan(density_values)):
            raise ValueError("densities must not contain NaNs")

    def plot(self,  l_values, mlt_values, density_values, date,
             fig_size=(7, 6.5), title_coordinates=(math.pi / 2 + 0.45, 8)):

        self._check_inputs(density_values)

        self._set_date(date)

        self._set_title_coordinates(title_coordinates)

        self._set_colour_map()

        self._set_figure(plt.figure(figsize=fig_size))

        self._set_ax(self._get_figure().add_subplot(1, 1, 1, projection='polar'))

        angle_values = PlasmaspherePlot._mlt_to_angle(mlt_values)

        self._plot_plasma_density(
            angle_values,
            l_values,
            density_values
        )

        self._draw_earth_night_side()
        self._draw_earth_day_side()

        self._set_style()

        self._get_ax().text(
            x=self._get_title_coordinates()[0],
            y=self._get_title_coordinates()[1],
            s="{}".format(self._get_date().strftime("%Y-%m-%d, %H:%M")),
            fontdict={'fontsize': 20, 'fontweight': "bold"})

        self._add_colour_bar()

        self._get_figure().subplots_adjust(left=0.05, right=0.85)

    def save(self, path):
        plt.savefig(path)
        plt.close()

