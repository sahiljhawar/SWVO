import sys
import datetime as dt
import os
import inspect
import pandas as pd
import numpy as np

LOCAL_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(LOCAL_PATH, "../../../../"))
from data_management.io.wp2.read_swift import SwiftReader


class TestReadSWIFT(object):

    def test_init_standard(self):
        try:
            SwiftReader()
            assert True
        except FileNotFoundError:
            assert False

    def test_init_no_data_folder(self):
        try:
            SwiftReader(wp2_output_folder="/FAKE_FOLDER/")
            assert False
        except FileNotFoundError:
            assert True

    def test_read_no_args(self):
        reader = SwiftReader()
        try:
            data_gsm, data_hgc = reader.read(date=None, fields=None)
            assert isinstance(data_gsm, pd.DataFrame)
            assert isinstance(data_hgc, pd.DataFrame)
        except FileNotFoundError:
            pass

    def test_read_all_fields(self):
        reader = SwiftReader()
        data_gsm, data_hgc = reader.read(date=dt.datetime(2021, 4, 7), fields=None)
        assert isinstance(data_gsm, pd.DataFrame)
        assert isinstance(data_hgc, pd.DataFrame)
        for f in SwiftReader.DATA_FIELDS:
            assert f in data_gsm
            assert f in data_hgc

    def test_read_some_fields(self):
        reader = SwiftReader()
        fields = np.random.choice(SwiftReader.DATA_FIELDS, np.random.randint(1, len(SwiftReader.DATA_FIELDS)),
                                  replace=False)
        data_gsm, data_hgc = reader.read(date=dt.datetime(2021, 4, 7), fields=fields)
        for f in fields:
            assert f in data_gsm
            assert f in data_hgc

        not_fields = [f for f in SwiftReader.DATA_FIELDS if f not in fields]
        for f in not_fields:
            assert f not in data_gsm
            assert f not in data_hgc

    def test_fake_fields(self):
        reader = SwiftReader()
        fields = ["no_field", "funny_field"]
        try:
            reader.read(date=dt.datetime(2021, 4, 7), fields=fields)
            assert False
        except KeyError:
            assert True
