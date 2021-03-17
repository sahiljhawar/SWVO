import sys
from os import path

import math
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import matplotlib.ticker as mticker

from datetime import datetime

import warnings

basepath = path.dirname(__file__)
sys.path.append(path.abspath(path.join(basepath, "..", "..", "..")))
from plotting.plotting_base import PlotOutput


class PlasmaspherePlot(PlotOutput):

    def _draw_earth_night_side(self):
        """
        It draws the night side of the earth

        :param ax:
        :return:
        """
        self._get_ax().fill_between(np.linspace(-np.pi / 2, np.pi / 2, 100),
                                    0, 1, alpha=1, color='k')

    def _set_axes_ticks_labels(self):

        ticks_loc = [x for x in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
        self._get_ax().xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        self._get_ax().set_xticklabels(
            labels=[xlabel for xlabel in np.arange(0, 24, 3)]
        )

        print([xlabel for xlabel in np.arange(0, 24, 3)])

        ticks_loc = [1, 2, 3, 4, 5, 6]
        self._get_ax().yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        self._get_ax().set_ylim(0, 6.5)
        self._get_ax().set_yticklabels(
            labels=ticks_loc
        )
        self._get_ax().grid(c='grey', lw=0.5, ls='-', visible=True)


    def _plot_plasma_density(self,
                             angle_values,
                             l_values,
                             density_values
                             ):
        self._get_ax().scatter(angle_values,
                               l_values,
                               c=density_values,
                               s=l_values ** 2,
                               cmap=self._get_colour_map(),
                               alpha=0.4
                               )

    def _set_colour_bar_axes(self, axes):
        self.colour_bar_axes = axes

    def _get_colour_bar_axes(self):
        return self.colour_bar_axes

    def _add_colour_bar(self):

        colour_bar_normalization = mpl.colors.Normalize(vmin=0,
                                                        vmax=3.99
                                                        )

        color_bar = plt.colorbar(mpl.cm.ScalarMappable(
            norm=colour_bar_normalization,
            cmap=self._get_colour_map()
        ),
            ax=self._get_ax(),
            use_gridspec=True,
            shrink=0.5,
            pad=0.15
        )
        color_bar.set_label('$log_{10}(n_e)$')

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
              fig_size=(4, 4)):

        if not isinstance(date, datetime):
            raise ValueError("date must be datetime object")

        self._set_date(date)
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),
                               figsize=fig_size)
        self._set_figure(fig)
        self._set_ax(ax)

        self._check_inputs(density_values)

        self._set_colour_map()

        angle_values = PlasmaspherePlot._mlt_to_angle(mlt_values)
        self._plot_plasma_density(
            angle_values,
            l_values,
            density_values,
        )

        self._draw_earth_night_side()

        self._set_axes_ticks_labels()

        self._get_ax().set_title(
            "{}".format(self._get_date().strftime("%Y-%m-%d, %H:%M")),
        fontdict={'fontweight': 'bold'})

        self._add_colour_bar()

        self._get_figure().subplots_adjust(left=0.07, right=0.8)
        plt.tight_layout()

    def save(self, path):
        plt.savefig(path)
        plt.close()

    @staticmethod
    def plot_output(data):

        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas dataframe")

        required_columns = ["L", "MLT", "predicted_densities", "date"]
        for column in required_columns:
            if column not in data.columns:
                raise ValueError("column {} is missing".format(column))

        if not isinstance(data.iloc[0]["date"], datetime):
            raise ValueError("values of date column must be datetime objects")

        l_values = data["L"]
        mlt_values = data["MLT"]
        density_values = data["predicted_densities"]
        date = data.iloc[0]["date"]

        plotter = PlasmaspherePlot()
        plotter.plot(l_values, mlt_values, density_values, date)
