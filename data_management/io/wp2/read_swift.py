from data_management.io.base_file_reader import BaseReader
import json
import datetime as dt
import glob
import numpy as np
import pandas as pd
import os


class SwiftReader(BaseReader):
    PROT_MASS = 1.67262192369e-27

    def __init__(self, date_folder="/PAGER/WP2/data/outputs/"):
        super().__init__()
        self.data_folder = os.path.join(date_folder, "SWIFT")

    @staticmethod
    def read_single_file(file_name, fields=None):
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
        n = np.array(data["arrays"]["Rho"]["data"]) / SwiftReader.PROT_MASS * 1.0e-6

        df = pd.DataFrame({"proton_density": n, "speed": speed, "b": b, "temperature": temperature,
                           "bx": bx, "by": by, "bz": bz}, index=time)
        if fields is not None:
            df = df[fields]
        return df

    def read(self, date=None, fields=None):
        if date is None:
            date = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        date_to_string = date.strftime("%Y%m%d")
        gsm_file = glob.glob(self.data_folder + date_to_string + "*/gsm*")[0]
        hgc_file = glob.glob(self.data_folder + date_to_string + "*/hgc*")[0]

        data_gsm = SwiftReader.read_single_file(gsm_file)
        data_hgc = SwiftReader.read_single_file(hgc_file)
        return data_gsm, data_hgc
