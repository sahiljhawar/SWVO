from unittest import mock
import os
import inspect
import sys
import pandas as pd
import numpy as np
import datetime as dt

LOCAL_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(LOCAL_PATH, "../../../../"))
from data_management.io.wp3.read_kp import KPReader


class TestReadSWIFT(object):

    @mock.patch("data_management.io.wp3.read_kp.KPReader._check_data_folder", return_value=None, autospec=True)
    def test_init_folder_found(self, mocker):
        try:
            KPReader()
            assert True
        except FileNotFoundError:
            assert False

    @mock.patch("data_management.io.wp3.read_kp.KPReader._check_data_folder", side_effect=FileNotFoundError(),
                autospec=True)
    def test_init_folder_not_found(self, mocker):
        try:
            KPReader()
            assert False
        except FileNotFoundError:
            assert True

    KP = pd.DataFrame(index=pd.date_range(start="2021-01-01", end="2021-01-01 23:59", freq="3H"))
    KP["kp"] = np.random.randint(0, 9, len(KP))

    @mock.patch("data_management.io.wp3.read_kp.KPReader._read_single_file",
                return_value=(KP, dt.datetime(2021, 1, 1)), autospec=True)
    @mock.patch("data_management.io.wp3.read_kp.KPReader._check_data_folder", return_value=None, autospec=True)
    def test_read_no_date_file_found(self, mocker, mocker2):
        reader = KPReader()
        for source in ["niemegk", "swpc", "l1", "swift"]:
            data, timestamp_data = reader.read(source, requested_date=None)
            assert data.equals(self.KP)
            assert isinstance(timestamp_data, dt.datetime)

    @mock.patch("data_management.io.wp3.read_kp.KPReader._read_single_file",
                return_value=(KP, dt.datetime(2021, 1, 1)), autospec=True)
    @mock.patch("data_management.io.wp3.read_kp.KPReader._check_data_folder", return_value=None, autospec=True)
    def test_read_wrong_source(self, mocker, mocker2):
        reader = KPReader()
        try:
            reader.read("fake_source", requested_date=None)
            assert False
        except RuntimeError:
            assert True
