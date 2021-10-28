import os
import logging
import pandas as pd
import datetime as dt

from data_management.io.base_file_reader import BaseReader


class PlasmaspherePredictionReader(BaseReader):

    def __init__(self, wp3_output_folder="/PAGER/WP3/data/outputs/"):
        super().__init__()
        self.wp3_output_folder = wp3_output_folder
        self.file = None
        self.requested_date = None

    def _check_data_folder(self):
        if not os.path.exists(self.wp3_output_folder):
            msg = "Data folder for WP3 KP output not found...impossible to retrieve data."
            logging.error(msg)
            raise FileNotFoundError(msg)

    @staticmethod
    def _read_single_file(folder, date):
        """
        It reads the plasmasphere prediction from the specified folder.
        If self.file is None and self.requested_date is not None,
        looks for the most recent file having the self.requested_date.
        If  self.file is not None and self.requested_date is None, returns
        the full data contained in self.file


        :param folder: folder where we look for the plasmasphere prediction
        :type folder: str
        :return: the plasmasphere prediction for the requested date
        :rtype: pandas.DataFrame
        :raises: ValueError if self.file is not None, but it cannot be found.
                 RuntimeError is self.file is None, and no files containing
                 requested_date can be found.
        """

        file_name = "plasmasphere_density_{}-{}-{}-{}-{}.csv".format(date.year,
                                                                     str(date.month).zfill(2),
                                                                     str(date.day).zfill(2),
                                                                     str(date.hour).zfill(2),
                                                                     str(date.minute).zfill(2))

        file_path = os.path.join(folder, file_name)

        if not os.path.isfile(file_path):
            msg = "No suitable files found in the folder {} for the requested date {}".format(folder, date)
            logging.warning(msg)
            return None

        data = pd.read_csv(file_path, parse_dates=["date"])
        data["t"] = data["date"]
        data.drop(labels=["date"], axis=1, inplace=True)
        return data

    def read(self, source, requested_date=None) -> pd.DataFrame:
        """
        Reads one of the available PAGER plasmasphere density prediction.

        :param source: The source of plasmasphere density product requested. Available only "gfz_plasma".
        :type source: str
        :param requested_date: Date of plasma density prediction thar we want to read up to hour precision.
        :type requested_date: datetime.datetime or None

        :raises: RuntimeError if the sources of data requested is not among the available ones.

        :return: pandas.DataFrame with L, MLT, density and date as columns
        """

        if requested_date is None:
            requested_date = dt.datetime.utcnow().replace(microsecond=0, minute=0, second=0)

        if source == "gfz_plasma":
            requested_date = requested_date.replace(minute=0)
            return self._read_single_file(os.path.join(self.wp3_output_folder, "GFZ_PLASMA"), requested_date)
        else:
            msg = "Source {} requested for reading plasmasphere prediction not available...".format(source)
            logging.error(msg)
            raise RuntimeError(msg)
