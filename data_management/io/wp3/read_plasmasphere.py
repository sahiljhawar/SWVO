import os
import logging
import glob

import numpy as np
import pandas as pd

from datetime import datetime

from data_management.io.base_file_reader import BaseReader


class PlasmaspherePredictionReader(BaseReader):

    def __init__(self, data_folder="/PAGER/WP3/data/outputs/"):
        super().__init__()
        self.data_folder = data_folder
        self.file=None
        self.requested_date=None

    @staticmethod
    def _get_file_full_path(directory, file_name):
        """
        If it founds the file, it returns the path,
        otherwise it raises.

        :param directory: string
        :param file_name: string

        :raises: if it cannot find the file

        :return: string
        """

        complete_file_path = directory + file_name
        path_list = glob.glob(complete_file_path)
        if len(path_list) > 0:
            return path_list[0]
        else:
            return None

    @staticmethod
    def _raise_if_date_not_present(df, date):
        df_date = df[df["date"] == date]
        if df_date.empty:
            raise ValueError("date {} is not present".format(date))

    @staticmethod
    def _get_date_components(date):
        year = str(date.year)
        month = date.strftime('%m')
        day = date.strftime('%d')
        hour = date.strftime('%H')
        minute = date.strftime("%M")
        return year, month, day, hour, minute

    @staticmethod
    def _is_date_present(file_full_path, date):
        df_file = pd.read_csv(file_full_path,
                              parse_dates=["date"])
        df_date = df_file[df_file["date"] == date]
        if df_date.empty:
            return False
        else:
            return True

    def _find_file(self, folder):

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
                PlasmaspherePredictionReader._get_full_path(folder,
                                                            self.file)
            if file_full_path is None:
                raise ValueError(
                "file {} doesn't exist in the directory {}".format(self.file,
                                                                   folder)
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
                                   "containing the date {}".format(folder, date))
            df_file = pd.read_csv(file_full_path,
                                  parse_dates=["date"])

            return df_file[df_file["date"] == self.date]

    def read(self, source, requested_date, file=None):
        """
        Reads one of the available PAGER plasmasphere density prediction.

        :param source: The source of plasmasphere density product requested.
                        Choose among "GFZ_PLASMA"
        :param requested_date: Requested data for which we want to have the
                               plasmadensity prediction.
                               datetime instance which needs up to hour precision,
                               since at this resolution the plasmasphere is predicted.
        :param file: specifies a file from which to read the prediction.
                     If None it will read from the most recent file in which
                     the date is present, since it gives the most accurate prediction.
                     If not None, it is a string specifying the file name.

        :raises: ValueError if requested_date is not in datetime format
        :raises: RuntimeError if the sources of data requested is not among the available ones.

        :return: an instance of pandas.DataFrame format having L, MLT, density and date as columns
        """

        self.file = file

        if not isinstance(requested_date, datetime):
            raise ValueError("requested_date must be a datetime variable")

        if requested_date.minute !=0 or requested_date.second !=0 or \
                requested_date.microsecond != 0:
            requested_date = requested_date.replace(minute=0,
                                                    second=0,
                                                    microsecond=0)
        self.requested_date = requested_date


        if source == "gfz_plasma":
            return self._read_from_source(os.path.join(self.data_folder, "GFZ_PLASMA/*"), requested_date)
        else:
            msg = "Source {} requested for reading plasmasphere prediction not available...".format(source)
            logging.error(msg)
            # save the logs?????????
            # or the logs outside?????
            raise RuntimeError(msg)

