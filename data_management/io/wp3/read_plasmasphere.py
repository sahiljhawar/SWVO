import os
import logging
import glob

import numpy as np
import pandas as pd

from datetime import datetime
import datetime as dt

from data_management.io.base_file_reader import BaseReader


class PlasmaspherePredictionReader(BaseReader):

    def __init__(self, wp3_output_folder="/PAGER/WP3/data/outputs/"):
        super().__init__()
        self.wp3_output_folder = wp3_output_folder
        self.file = None
        self.requested_date = None

    @staticmethod
    def _get_file_full_path(directory, file_name):
        """
        If it founds the file, it returns the full path of the file,
        otherwise it returns None.

        :param directory: directory where the file is
        :type directory: str
        :param file_name: name of the file
        :type file_name: str

        :return: file full path or None
        :rtype: string or None
        """

        complete_file_path = directory + file_name
        path_list = glob.glob(complete_file_path)
        if len(path_list) > 0:
            return path_list[0]
        else:
            return None

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

    @staticmethod
    def _is_date_present(file_full_path, date):
        """
        It checks whether the date is contained in the specified file

        :param file_full_path: full path of a file containing plasma density
        :type file_full_path: str
        :param date: date for which we want the plasmadensity
        :type date:
        :return: indicator which tells whether the date is present or not
                 in the file
        :rtype: boolean
        """
        df_file = pd.read_csv(file_full_path,
                              parse_dates=["date"])
        df_date = df_file[df_file["date"] == date]
        if df_date.empty:
            return False
        else:
            return True

    def _get_file_path(self, folder):
        """
        It returns the file in the specified folder in which self.requested_date
        is present. If it cannot find the file, it returns None

        :param folder: it specifies the folder where to look
                       for the file
        :type folder: str
        :return: file in which self.requested_date is present or None
        :rtype: string or None
        """

        year, month, day, hour, minute = \
            PlasmaspherePredictionReader._get_date_components(self.requested_date)
        file_name = "plasmasphere_density_{}-{}-{}-{}-{}.csv".format(
            year, month, day, hour, minute
        )

        return PlasmaspherePredictionReader._get_file_full_path(folder,
                                                                file_name)

    def _read_from_folder(self, folder):
        """
        It reads the plasmasphere prediction from the specified folder.
        If self.file is None and self.requested_date is not None,
        looks for the most recent file having the self.requested_date.
        If  self.file is not None and self.requested_date is None, returns
        the full data contained in self.file


        :param folder: folder where we look for the plasmasphere prediction
        :type folder: str
        :return: the plasmasphere prediction for the requested date
        :rtype: instance of pd.DataFrame
        :raises: ValueError if self.file is not None, but it cannot be found.
                 RuntimeError is self.file is None, and no files containing
                 requested_date can be found.
        """


        file_full_path = self._get_file_path(folder)
        if file_full_path is None:
            msg = "No suitable files found in the folder {} " \
                  "for the requested date {}".format(folder,
                                                     self.requested_date)
            logging.error(msg)
            raise FileNotFoundError(msg)
        else:
            return pd.read_csv(file_full_path,
                               parse_dates=["date"])



    def read(self, source, requested_date) -> pd.DataFrame:
        """
        Reads one of the available PAGER plasmasphere density prediction.

        :param source: The source of plasmasphere density product requested.
                        Available only "gfz_plasma".
        :type source: str
        :param requested_date: Requested data for which we want to read the
                               plasma density data. It needs to be up to
                               hour precision since the plasmasphere is
                               predicted with this time resolution.
        :type requested_date: datetime.datetime

        :raises: ValueError if requested_date is not in datetime format
        :raises: RuntimeError if the sources of data requested is not among
                 the available ones.

        :return: an instance of pandas.DataFrame format having L, MLT, density
                 and date as columns
        """

        self.requested_date = requested_date

        if source == "gfz_plasma":
            return self._read_from_folder(os.path.join(self.wp3_output_folder,
                                                       "GFZ_PLASMA/*")
                                          )
        else:
            msg = "Source {} requested for reading plasmasphere prediction " \
                  "not available...".format(source)
            logging.error(msg)
            raise RuntimeError(msg)
