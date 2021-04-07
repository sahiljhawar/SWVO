from data_management.io.base_file_reader import BaseReader
import json
import datetime as dt
import glob
import numpy as np
import pandas as pd
import os
import logging


class SwiftReader(BaseReader):
    """
    This class reads data from PAGER WP2 SWIFT outputs. It has only one public method used
    to read the output produced on a given date. It assumes that SWIFT runs only once a day
    which is the current way in which the SWIFT software is configured.
    """
    PROTON_MASS = 1.67262192369e-27
    DATA_FIELDS = ["proton_density", "speed", "b", "temperature", "bx", "by", "bz"]

    def __init__(self, wp2_output_folder="/PAGER/WP2/data/outputs/"):
        """
        :param wp2_output_folder: The path to the output folder of WP2 products.
        :type wp2_output_folder: str
        """
        super().__init__()
        self.data_folder = os.path.join(wp2_output_folder, "SWIFT/")

    @staticmethod
    def _read_single_file(file_name, fields=None) -> pd.DataFrame:
        """
        This function reads one the two available json file of SWIFT output and extract relevant variables
        combining them into a pandas DataFrame.

        The path to the file to read.
        :param file_name: The path of the file to read.
        :type file_name: str
        :param fields: Lists of fields to extract from the DataFrame. The list needs to contain a subset
                       of available fields. if None, all the fields available are retrieved.
        :type fields: list
        :return: A pandas.DataFrame with requested variables.
        """
        with open(file_name) as f:
            data = json.load(f)

        time = list(map(lambda x: dt.datetime.utcfromtimestamp(int(x)), data["arrays"]["Unix time"]["data"]))

        ux = np.array(data["arrays"]["Vr"]["data"]) * np.sin(
            data["arrays"]["Vtheta"]["data"]) * np.cos(data["arrays"]["Vphi"]["data"]) / 1000.0
        uy = np.array(data["arrays"]["Vr"]["data"]) * np.sin(
            data["arrays"]["Vtheta"]["data"]) * np.sin(data["arrays"]["Vphi"]["data"]) / 1000.0
        uz = np.array(data["arrays"]["Vr"]["data"]) * np.cos(data["arrays"]["Vtheta"]["data"]) / 1000.0

        bx = np.array(data["arrays"]["Br"]["data"]) * np.sin(
            data["arrays"]["Btheta"]["data"]) * np.cos(data["arrays"]["Bphi"]["data"]) * 1.0e9
        by = np.array(data["arrays"]["Br"]["data"]) * np.sin(
            data["arrays"]["Btheta"]["data"]) * np.sin(data["arrays"]["Bphi"]["data"]) * 1.0e9
        bz = np.array(data["arrays"]["Br"]["data"]) * np.cos(data["arrays"]["Btheta"]["data"]) * 1.0e9

        temperature = np.array(data["arrays"]["Temperature_ion"]["data"])
        speed = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
        b = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
        n = np.array(data["arrays"]["Rho"]["data"]) / SwiftReader.PROTON_MASS * 1.0e-6

        df = pd.DataFrame({"proton_density": n, "speed": speed, "b": b, "temperature": temperature,
                           "bx": bx, "by": by, "bz": bz}, index=time)
        if fields is not None:
            df = df[fields]
        return df

    def read(self, date=None, fields=None) -> (pd.DataFrame, pd.DataFrame):
        """
        This function reads output data from SWIFT and returns it in the form of a tuple of two pandas dataframe,
        each for each coordinate system available, GSM and HGC.

        :param date: The date in which data has been produced. It assumes that the data is produced once a day. If None
                     the data with current date is requested.
        :type date: datetime.datetime or None
        :param fields: List of fields to be extracted from the available data.
        :type fields: list or None
        :raises: RuntimeError: when a field requested is not among available field list or when the file requested
                               with output data is not found.
        :return: Tuple of GSM and HGC data as pandas data frames
        """
        if date is None:
            date = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        date_to_string = date.strftime("%Y%m%d")

        if fields is not None:
            for f in fields:
                if f not in SwiftReader.DATA_FIELDS:
                    msg = "Requested field from SWIFT data not available..."
                    logging.error(msg)
                    raise RuntimeError(msg)

        try:
            gsm_file = glob.glob(os.path.join(self.data_folder, date_to_string + "*/gsm*"))[0]
            data_gsm = SwiftReader._read_single_file(gsm_file, fields)
        except IndexError:
            msg = "GSM SWIFT output file for date {} not found...impossible to read".format(date_to_string)
            logging.error(msg)
            raise RuntimeError(msg)

        try:
            hgc_file = glob.glob(os.path.join(self.data_folder, date_to_string + "*/hgc*"))[0]
            data_hgc = SwiftReader._read_single_file(hgc_file, fields)
        except IndexError:
            msg = "GSM SWIFT output file for date {} not found...impossible to read".format(date_to_string)
            logging.error(msg)
            raise RuntimeError(msg)

        return data_gsm, data_hgc
