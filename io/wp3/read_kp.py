import sys
sys.path.append("/PAGER/WP8/data_management/io/")
from base_file_reader import BaseReader

import datetime as dt
import glob
import pandas as pd
import os
import logging


class KPReader(BaseReader):

    def __init__(self):
        super().__init__()
        self.data_folder = "/PAGER/WP3/data/outputs/"

    @staticmethod
    def _read_single_file(folder, requested_date=None):
        start_date = dt.datetime(1900, 1, 1)
        last_file = None

        if requested_date is None:
            for file in glob.glob(folder):
                date = file.split("/")[-1]
                date = date.split(".")[0]
                date = dt.datetime.strptime(date.split("_")[-1], "%Y%m%d")

                if date > start_date:
                    last_file = file
                    start_date = date
        else:
            for file in glob.glob(folder):
                date = file.split("/")[-1]
                date = date.split(".")[0]
                try:
                    date = dt.datetime.strptime(date.split("_")[-1], "%Y%m%d")
                except ValueError:
                    date = dt.datetime.strptime(date.split("_")[-1], "%Y-%m-%d")
                if date == requested_date:
                    last_file = file
                    start_date = date
                    break
        try:
            df = pd.read_csv(last_file, names=["t", "kp"])
            df["t"] = pd.to_datetime(df["t"])
            df.index = df["t"]
            df["index"] = ["kp"] * len(df)
            return df, start_date
        except FileNotFoundError:
            logging.error("File not found in folder {}...".format(folder))
            return None, None
        except ValueError:
            logging.error("No file found for requested date {}".format(requested_date))
            return None, None

    def read(self, source, requested_date=None):
        """
        Reads one of the available PAGER Kp forecast products.

        :param source: The source of Kp product requested. Choose among "niemegk", "swift", "swpc" and "l1"
        :type source: str
        :param requested_date:
        :type requested_date: datetime.datetime or None
        :return:
        """

        if source == "niemegk":
            return self._read_single_file(os.path.join(self.data_folder, "NIEMEGK/*"), requested_date)
        elif source == "swpc":
            return self._read_single_file(os.path.join(self.data_folder, "SWPC/*"), requested_date)
        elif source == "l1":
            return self._read_single_file(os.path.join(self.data_folder, "RBM9/*"), requested_date)
        elif source == "swift":
            return self._read_single_file(os.path.join(self.data_folder, "SWIFT/*"), requested_date)
        else:
            msg = "Source {} requested for reading Kp not available...".format(source)
            logging.error(msg)
            raise RuntimeError(msg)
