import os
import logging
import glob

import numpy as np
import pandas as pd

from datetime import datetime
import datetime as dt

from data_management.io.base_file_reader import BaseReader


class PlasmaspherePredictionReader(BaseReader):

    def __init__(self, data_folder="/PAGER/WP3/data/outputs/"):
        super().__init__()
        self.data_folder = data_folder
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

        :return: string or None
        """

        complete_file_path = directory + file_name
        path_list = glob.glob(complete_file_path)
        if len(path_list) > 0:
            return path_list[0]
        else:
            return None

    @staticmethod
    def _raise_if_date_not_present(df, date):
        """
        It raises if date is not contained in df["date"] column

        :param df: dataframe containing plasma density at different times
        :type df: instance of pd.DataFrame
        :param date: date for which we want the plasmadensity
        :type date:
        :raises: ValueError if date is not present in  df["date"] column
        """
        df_date = df[df["date"] == date]
        if df_date.empty:
            raise ValueError("date {} is not present".format(date))

    @staticmethod
    def _get_date_components(date):
        """
        It gets a datetime instance and returns year, month, day, hour, minute

        :param date: a date
        :type date: an instance of datetime object
        :return: tuple of int, year, month, day, hour, minute
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
        :return: True or False
        """
        df_file = pd.read_csv(file_full_path,
                              parse_dates=["date"])
        df_date = df_file[df_file["date"] == date]
        if df_date.empty:
            return False
        else:
            return True

    def _find_file(self, folder):
        """
        It returns the file in the specified folder in which self.date
        is present. If it cannot find the file, it returns None

        :param folder: it specifies the folder where to look
                       for the file
        :type folder: str
        :return: string or None
        """

        file_full_path = None
        dates = np.array([self.date - datetime.timedelta(hours=hours_to_shift)
                          for hours_to_shift in range(48)])

        for date in dates:

            year, month, day, hour, minute = \
                PlasmaspherePredictionReader._get_date_components(date)
            file_name = "plasmasphere_density_{}-{}-{}-{}-{}.csv".format(
                year, month, day, hour, minute
            )
            file_full_path = \
                PlasmaspherePredictionReader._get_file_full_path(folder,
                                                                 file_name)
            if file_full_path is not None:
                if self._is_date_present(file_full_path, date):
                    break

        return file_full_path

    def _read_from_source(self, folder, requested_date):
        """
        It reads the plasmasphere prediction for the requested date
        from the specified folder.

        :param folder: folder where we look for the plasmasphere prediction
        :type folder: str
        :param requested_date: date for which we want the plasmasphere
                               prediction
        :type requested_date:
        :return: instance of pd.DataFrame containing the plasmasphere
                 prediction for the requested date
        :raises: ValueError if self.file is not None, but it cannot be found.
                 RuntimeError is self.file is None, and no files containing
                 requested_date can be found.
        """

        for file in glob.glob(folder):
            date = file.split("/")[-1]
            date = date.split(".")[0]
            try:
                date = datetime.strptime(date.split("_")[-1], "%Y%m%d")
            except ValueError:
                date = datetime.strptime(date.split("_")[-1], "%Y-%m-%d")
            if date == requested_date:
                last_file = file
                start_date = date
                break

        if self.file is not None:

            file_full_path = \
                PlasmaspherePredictionReader._get_file_full_path(folder,
                                                                 self.file)
            if file_full_path is None:
                raise ValueError(
                    "file {} doesn't exist in the directory {}".format(
                        self.file,
                        folder
                    )
                )

            df_file = pd.read_csv(file_full_path,
                                  parse_dates=["date"])
            PlasmaspherePredictionReader._raise_if_date_not_present(df_file,
                                                                    self.date)
            return df_file[df_file["date"] == self.date]

        else:

            file_full_path = self._find_file(folder)
            if file_full_path is None:
                raise RuntimeError("No suitable files found in the folder {}"
                                   "containing the date {}".format(folder,
                                                                   date))
            df_file = pd.read_csv(file_full_path,
                                  parse_dates=["date"])

            return df_file[df_file["date"] == self.date]

    def read(self, source, requested_date=None, file=None) -> pd.DataFrame:
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
        :param file: specifies a file from which to read the prediction.
                     If None it will read from the most recent file in which
                     the date is present, since it gives the most accurate
                     prediction. If not None, it is a string specifying the
                     file name.

        :raises: ValueError if requested_date is not in datetime format
        :raises: RuntimeError if the sources of data requested is not among
                 the available ones.

        :return: an instance of pandas.DataFrame format having L, MLT, density
                 and date as columns
        """

        self.file = file

        if requested_date is None:
            requested_date = dt.datetime.utcnow().replace(minute=0,
                                                          second=0,
                                                          microsecond=0)

        if not isinstance(requested_date, datetime):
            raise ValueError("requested_date must be a datetime variable")

        requested_date = requested_date.replace(minute=0,
                                                second=0,
                                                microsecond=0)
        self.requested_date = requested_date

        if source == "gfz_plasma":
            return self._read_from_source(os.path.join(self.data_folder,
                                                       "GFZ_PLASMA/*"),
                                          requested_date)
        else:
            msg = "Source {} requested for reading plasmasphere prediction " \
                  "not available...".format(source)
            logging.error(msg)
            raise RuntimeError(msg)
