import datetime as dt
import glob
import pandas as pd
import os
import logging

from data_management.io.base_file_reader import BaseReader



class Hp60Reader(BaseReader):
    """
    Reader class for Kp products from WP3 PAGER project. It reads Hp60 forecasts.
    """

    def __init__(self, hp60_output_folder):
        """
        :param hp60_output_folder: The path to data outputs for hp60.
        :type hp60_output_folder: str
        """
        super().__init__()
        self.data_folder = hp60_output_folder
        self._check_data_folder()

    def _check_data_folder(self):
        if not os.path.exists(self.data_folder):
            msg = "Data folder for HP60 output not found...impossible to retrieve data."
            logging.error(msg)
            raise FileNotFoundError(msg)

    @staticmethod
    def _correct_column_name_for_files_generated_before_2023(df):
        return df.rename(columns={"kp": "Hp60"})

    def read(self, requested_date=None, model_name="HP60-FULL-SW-SWAMI-PAGER", header=False) -> tuple:
        """
        This function reads one of the available PAGER Hp forecast products.

        :param requested_date: Requested data for data to read. If None it reads data from the latest file produced.
        :type requested_date: datetime.datetime or None
        :param model_name:
        :type model_name: str or None
        :param header:
        :type header: bool
        :raises: RuntimeError: This exception is raised if the sources of data requested is not among
                 the available ones.

        :return: tuple of data in pandas.DataFrame format and datetime.datetime of the date extracted from the file.
        """
        file_to_read = None
        if requested_date is None:
            requested_date = dt.datetime.utcnow().replace(microsecond=0, minute=0, second=0)

        date_found = None
        files = sorted(glob.glob(self.data_folder + "/*"))
        for file in files:
            date = file.split("/")[-1]
            date = date.split(".")[0]
            target_date = requested_date
            date = dt.datetime.strptime(date.split("_")[-1], "%Y%m%dT%H%M%S")
            if target_date == date:
                if model_name not in file:
                    continue
                file_to_read = file
                date_found = date

        if file_to_read is None:
            logging.error("File not found in folder {} for the date {} and "
                          "model name {}".format(self.data_folder,
                                                 requested_date,
                                                 model_name))
            return None, None

        if not header:
            df = pd.read_csv(file_to_read, names=["t", "Hp60"])
        else:
            df = pd.read_csv(file_to_read)
            if "kp" in df.columns:
                df = Hp60Reader._correct_column_name_for_files_generated_before_2023(df)

        df["t"] = pd.to_datetime(df["t"])
        df.index = df["t"]
        df.drop(labels=["t"], axis=1, inplace=True)

        return df, date_found


