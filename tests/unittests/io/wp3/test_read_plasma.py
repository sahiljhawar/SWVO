from unittest import mock
import os
import inspect
import sys
import pandas as pd
import numpy as np
import datetime as dt

LOCAL_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(LOCAL_PATH, "../../../../"))
from data_management.io.wp3.read_plasmasphere import PlasmaspherePredictionReader


class TestReadPlasma(object):

    @mock.patch("data_management.io.wp3.read_plasmasphere.PlasmaspherePredictionReader._check_data_folder",
                return_value=None, autospec=True)
    def test_init_folder_found(self, mocker):
        try:
            PlasmaspherePredictionReader()
            assert True
        except FileNotFoundError:
            assert False

    @mock.patch("data_management.io.wp3.read_plasmasphere.PlasmaspherePredictionReader._check_data_folder",
                autospec=True, side_effect=FileNotFoundError())
    def test_init_folder_not_found(self, mocker):
        try:
            PlasmaspherePredictionReader()
            assert False
        except FileNotFoundError:
            assert True

    @mock.patch("os.path.exists", return_value=True, autospec=True)
    def test_check_data_folder_ok(self, mocker):
        reader = PlasmaspherePredictionReader()
        assert reader._check_data_folder() is None

    PLASMA = pd.DataFrame(index=pd.date_range(start="2021-01-01", end="2021-01-01 23:59", freq="3H"))
    PLASMA["L"] = np.random.uniform(1, 6, len(PLASMA))
    PLASMA["MLT"] = np.random.randint(0, 23, len(PLASMA))
    PLASMA["predicted_densities"] = np.random.uniform(1, 10, len(PLASMA))
    PLASMA["t"] = [dt.datetime(2022, np.random.randint(1, 12), np.random.randint(1, 28)) for i in range(len(PLASMA))]

    @mock.patch("data_management.io.wp3.read_plasmasphere.PlasmaspherePredictionReader._read_single_file",
                return_value=(PLASMA, dt.datetime(2021, 1, 1)), autospec=True)
    @mock.patch("data_management.io.wp3.read_plasmasphere.PlasmaspherePredictionReader._check_data_folder",
                return_value=None, autospec=True)
    def test_read_standard(self, mocker, mocker2):
        reader = PlasmaspherePredictionReader()
        data, timestamp_data = reader.read("gfz_plasma", requested_date=None)
        for field in ["L", "MLT", "t"]:
            assert field in data
        assert isinstance(timestamp_data, dt.datetime)
