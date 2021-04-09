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

    def test_init_folder_found(self):
        try:
            KPReader()
            assert True
        except FileNotFoundError:
            assert False

    def test_init_folder_not_found(self):
        try:
            KPReader("/FAKE_FOLDER/")
            assert False
        except FileNotFoundError:
            assert True

    def test_read_no_date(self):
        reader = KPReader()
        for source in ["niemegk", "swpc"]:
            data, timestamp_data = reader.read(source, requested_date=None)
            assert isinstance(data, pd.DataFrame)
            assert isinstance(timestamp_data, dt.datetime)
            assert "kp" in data
            if source == "swpc":
                assert dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) == timestamp_data
                #TODO Same does not work with Niemegk

    def test_read_specify_date_swpc(self):
        reader = KPReader()
        day = np.random.randint(1, 6)
        data, timestamp_data = reader.read("swpc", requested_date=dt.datetime(2021, 4, day))
        assert isinstance(data, pd.DataFrame)
        assert isinstance(timestamp_data, dt.datetime)
        assert timestamp_data == dt.datetime(2021, 4, day)
        assert "kp" in data

    def test_read_specify_date_niemegk(self):
        reader = KPReader()
        day = np.random.randint(1, 6)
        data, timestamp_data = reader.read("niemegk", requested_date=dt.datetime(2021, 4, day))
        assert isinstance(data, pd.DataFrame)
        assert isinstance(timestamp_data, dt.datetime)
        assert timestamp_data == dt.datetime(2021, 4, day)
        assert "kp" in data

    def test_read_specify_date_l1(self):
        reader = KPReader()
        day = np.random.randint(1, 6)
        data, timestamp_data = reader.read("l1", requested_date=dt.datetime(2021, 4, day))
        assert isinstance(data, pd.DataFrame)
        assert isinstance(timestamp_data, dt.datetime)
        assert timestamp_data == dt.datetime(2021, 4, day)
        assert (("kp" in data) or ("hp" in data))

    def test_read_specify_date_swift(self):
        reader = KPReader()
        day = np.random.randint(1, 6)
        data, timestamp_data = reader.read("swift", requested_date=dt.datetime(2021, 4, day))
        assert isinstance(data, pd.DataFrame)
        assert isinstance(timestamp_data, dt.datetime)
        assert timestamp_data == dt.datetime(2021, 4, day)
        assert "kp" in data

    def test_read_wrong_source(self):
        reader = KPReader()
        try:
            reader.read("fake_source", requested_date=None)
            assert False
        except RuntimeError:
            assert True
