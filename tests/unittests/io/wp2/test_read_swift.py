from unittest import mock
import os
import inspect
import sys
import pandas as pd
import numpy as np

LOCAL_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(LOCAL_PATH, "../../../../"))
from data_management.io.wp2.read_swift import SwiftReader


class TestReadSWIFT(object):

    @mock.patch("data_management.io.wp2.read_swift.SwiftReader._check_data_folder", return_value=None, autospec=True)
    def test_init_folder_found(self, mocker):
        try:
            SwiftReader()
            assert True
        except FileNotFoundError:
            assert False

    @mock.patch("data_management.io.wp2.read_swift.SwiftReader._check_data_folder", side_effect=FileNotFoundError(),
                autospec=True)
    def test_init_folder_not_found(self, mocker):
        try:
            SwiftReader()
            assert False
        except FileNotFoundError:
            assert True

    DATA_ALL_FIELDS = pd.DataFrame(index=pd.date_range(start="2021-01-01", end="2021-01-01 23:59", freq="1H"))
    for f in SwiftReader.DATA_FIELDS:
        DATA_ALL_FIELDS[f] = np.random.randint(0, 100, len(DATA_ALL_FIELDS))

    @mock.patch("data_management.io.wp2.read_swift.SwiftReader._read_single_file",
                return_value=DATA_ALL_FIELDS, autospec=True)
    @mock.patch("data_management.io.wp2.read_swift.SwiftReader._check_data_folder", return_value=None, autospec=True)
    @mock.patch("glob.glob", return_value=["name_of_file.json"], autospec=True)
    def test_read_no_date_no_fields_file_found(self, mocker, mocker2, mocker3):
        reader = SwiftReader()
        data_gsm, data_hgc = reader.read(date=None, fields=None)
        assert data_gsm.equals(self.DATA_ALL_FIELDS)
        assert data_hgc.equals(self.DATA_ALL_FIELDS)

    @mock.patch("data_management.io.wp2.read_swift.SwiftReader._read_single_file",
                return_value=DATA_ALL_FIELDS, autospec=True)
    @mock.patch("data_management.io.wp2.read_swift.SwiftReader._check_data_folder", return_value=None, autospec=True)
    @mock.patch("glob.glob", side_effect=FileNotFoundError(), autospec=True)
    def test_read_no_date_no_fields_file_not_found(self, mocker, mocker2, mocker3):
        reader = SwiftReader()
        try:
            reader.read(date=None, fields=None)
            assert False
        except FileNotFoundError:
            assert True

    @mock.patch("data_management.io.wp2.read_swift.SwiftReader._check_data_folder", return_value=None, autospec=True)
    @mock.patch("data_management.io.wp2.read_swift.SwiftReader._read_single_file",
                side_effect=KeyError(), autospec=True)
    def test_read_fake_fields(self, mocker, mocker2):
        reader = SwiftReader()
        try:
            reader.read(date=None, fields=["fake_field"])
            assert False
        except KeyError:
            assert True
