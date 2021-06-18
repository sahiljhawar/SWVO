import datetime as dt
import glob
import pandas as pd
import os
import logging

from data_management.io.base_file_reader import BaseReader


class KPReader(BaseReader):
    """
    Reader class for Kp products from WP3 PAGER project. It reads data from different sources of Kp data,
    e.g. SWPC forecast, SWIFT based and L1 real time based forecasts, as well as GFZ Niemegk nowcast.
    """

    def __init__(self, wp3_output_folder="/PAGER/WP3/data/outputs/"):
        """
        :param wp3_output_folder: The path to data outputs for WP3. It needs to contain sub-folders with individual
                                  products (e.g. SWPC, SWIFT).
        :type wp3_output_folder: str
        """
        super().__init__()
        self.data_folder = wp3_output_folder
        self._check_data_folder()

    def _check_data_folder(self):
        if not os.path.exists(self.data_folder):
            msg = "Data folder for WP3 KP output not found...impossible to retrieve data."
            logging.error(msg)
            raise FileNotFoundError(msg)

    @staticmethod
    def _read_single_file(folder, requested_date=None, header=False, model_name=None) -> tuple:
        """
        Reads a single file product with Kp data from PAGER.

        :param folder: The folder path of the file to read.
        :type folder: str
        :param requested_date: Requested date for data to read.
        :type requested_date: datetime.datetime
        :param header: True if the file product contains a header, False if not. (In the future we will uniform
                       the output formats and remove this parameter).
        :type header: bool

        :return: Tuple of data in pandas.DataFrame format and datetime.datetime of the date extracted from the file.
        """
        last_file = None
        if requested_date is None:
            start_date = dt.datetime(1900, 1, 1)
            for file in glob.glob(folder):
                date = file.split("/")[-1]
                date = date.split(".")[0]
                date = dt.datetime.strptime(date.split("_")[-1], "%Y%m%d")

                if date > start_date:
                    if (model_name is not None) and (model_name not in file):
                        continue
                    last_file = file
                    start_date = date
        else:
            start_date = requested_date
            time_delta = dt.timedelta(days=10000)
            for file in glob.glob(folder):
                date = file.split("/")[-1]
                date = date.split(".")[0]
                try:
                    date = dt.datetime.strptime(date.split("_")[-1], "%Y%m%d")
                except ValueError:
                    date = dt.datetime.strptime("_".join(date.split("_")[-2:]), "%Y%m%dT%H%M%S")
                if (requested_date >= date) and (requested_date - date < time_delta):
                    if (model_name is not None) and (model_name not in file):
                        continue
                    last_file = file
                    time_delta = requested_date - date

        try:
            if not header:
                # TODO Attention, this is not valid for hp or other indexes
                df = pd.read_csv(last_file, names=["t", "kp"])
            else:
                df = pd.read_csv(last_file)
            df["t"] = pd.to_datetime(df["t"])
            df.index = df["t"]
            df.drop(["t"], 1, inplace=True)
            return df, start_date
        except FileNotFoundError:
            logging.error("File not found in folder {}...".format(folder))
            return None, None
        except ValueError:
            logging.error("No file found for requested date {}".format(requested_date))
            return None, None

    def read(self, source, requested_date=None, model_name=None) -> tuple:
        """
        This function reads one of the available PAGER Kp forecast products.

        :param source: The source of Kp product requested. Choose among "niemegk", "swift", "swpc" and "l1"
        :type source: str
        :param requested_date: Requested data for data to read. If None it reads data from the latest file produced.
        :type requested_date: datetime.datetime or None
        :param model_name:
        :type model_name: str
        :raises: RuntimeError: This exception is raised if the sources of data requested is not among
                 the available ones.

        :return: tuple of data in pandas.DataFrame format and datetime.datetime of the date extracted from the file.
        """

        if source == "niemegk":
            data, data_timestamp = self._read_single_file(os.path.join(self.data_folder, "NIEMEGK/*"), requested_date)
        elif source == "swpc":
            data, data_timestamp = self._read_single_file(os.path.join(self.data_folder, "SWPC/*"), requested_date)
        elif source == "l1":
            data, data_timestamp = self._read_single_file(os.path.join(self.data_folder, "L1_FORECAST/*"),
                                                          requested_date, header=True, model_name=model_name)
        elif source == "swift":
            data, data_timestamp = self._read_single_file(os.path.join(self.data_folder, "SWIFT/*"),
                                                          requested_date, header=True)
        else:
            msg = "Source {} requested for reading Kp not available...".format(source)
            logging.error(msg)
            raise RuntimeError(msg)

        return data, data_timestamp
