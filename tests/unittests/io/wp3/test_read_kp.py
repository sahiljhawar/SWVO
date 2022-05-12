from unittest import mock
import pandas as pd
import numpy as np
import datetime as dt
import os

from data_management.io.wp3.read_kp import KPReader, KPEnsembleReader


class TestReadKp(object):

    @mock.patch("data_management.io.wp3.read_kp.KPReader._check_data_folder", return_value=None, autospec=True)
    def test_init_folder_found(self, mocker):
        folder = "/not/existing/path/"
        try:
            reader = KPReader(wp3_output_folder=folder)
            assert reader.data_folder == folder
        except FileNotFoundError:
            assert False

    @mock.patch("data_management.io.wp3.read_kp.KPReader._check_data_folder", side_effect=FileNotFoundError(),
                autospec=True)
    def test_init_folder_not_found(self, mocker):
        folder = "/not/existing/path/"
        try:
            KPReader(wp3_output_folder=folder)
            assert False
        except FileNotFoundError:
            assert True

    @mock.patch("os.path.exists", return_value=True, autospec=True)
    def test_check_data_folder_ok(self, mocker):
        folder = "/not/existing/path/"
        reader = KPReader(wp3_output_folder=folder)
        assert reader._check_data_folder() is None

    @mock.patch("os.path.exists", return_value=False, autospec=True)
    def test_check_data_folder_not_found(self, mocker):
        folder = "/not/existing/path/"

        try:
            reader = KPReader(wp3_output_folder=folder)
            reader._check_data_folder()
            assert False
        except FileNotFoundError:
            assert True

    KP = pd.DataFrame(index=pd.date_range(start="2021-01-01", end="2021-01-01 23:59", freq="3H"))
    KP["kp"] = np.random.randint(0, 9, len(KP))

    @mock.patch("data_management.io.wp3.read_kp.KPReader._read_single_file",
                return_value=(KP, dt.datetime(2021, 1, 1)), autospec=True)
    @mock.patch("data_management.io.wp3.read_kp.KPReader._check_data_folder", return_value=None, autospec=True)
    def test_read_standard(self, mocker, mocker2):
        folder = "/not/existing/path/"
        reader = KPReader(wp3_output_folder=folder)
        model_name = None
        for source in ["niemegk", "swpc", "l1"]:
            if source == "l1":
                model_name = "model_name"
            data, timestamp_data = reader.read(source, requested_date=None, model_name=model_name)
            assert data.equals(self.KP)
            assert isinstance(timestamp_data, dt.datetime)

    @mock.patch("data_management.io.wp3.read_kp.KPReader._read_single_file",
                return_value=(KP, dt.datetime(2021, 1, 1)), autospec=True)
    @mock.patch("data_management.io.wp3.read_kp.KPReader._check_data_folder", return_value=None, autospec=True)
    def test_read_standard(self, mocker, mocker2):
        folder = "/not/existing/path/"
        reader = KPReader(wp3_output_folder=folder)
        try:
            reader.read(source="l1", requested_date=None, model_name=None)
            assert False
        except AssertionError:
            assert True

    @mock.patch("data_management.io.wp3.read_kp.KPReader._read_single_file",
                return_value=(KP, dt.datetime(2021, 1, 1)), autospec=True)
    @mock.patch("data_management.io.wp3.read_kp.KPReader._check_data_folder", return_value=None, autospec=True)
    def test_read_wrong_source(self, mocker, mocker2):
        folder = "/not/existing/path/"
        reader = KPReader(wp3_output_folder=folder)
        try:
            reader.read("fake_source", requested_date=None)
            assert False
        except RuntimeError:
            assert True


class TestReadEnsembleKp(object):

    @mock.patch("data_management.io.wp3.read_kp.KPReader._check_data_folder", return_value=None, autospec=True)
    def test_init_folder_found(self, mocker):
        folder = "/not/existing/path/"
        sub_folder = "/not/existing/sub/folder/"
        try:
            reader = KPEnsembleReader(wp3_output_folder=folder, sub_folder=sub_folder)
            assert reader.data_folder == os.path.join(folder, sub_folder)
        except FileNotFoundError:
            assert False

    @mock.patch("data_management.io.wp3.read_kp.KPReader._check_data_folder", side_effect=FileNotFoundError(),
                autospec=True)
    def test_init_folder_not_found(self, mocker):
        folder = "/not/existing/path/"
        sub_folder = "/not/existing/sub/folder/"
        try:
            KPEnsembleReader(wp3_output_folder=folder, sub_folder=sub_folder)
            assert False
        except FileNotFoundError:
            assert True

    KP = pd.DataFrame(index=pd.date_range(start="2021-01-01", end="2021-01-01 23:59", freq="3H"))
    KP["kp"] = np.random.randint(0, 9, len(KP))

    DATA = [KP]*10
    DATE = dt.datetime(2021, 1, 1)

    @mock.patch("data_management.io.wp3.read_kp.KPEnsembleReader._read_ensemble_files",
                return_value=(DATA, DATE), autospec=True)
    @mock.patch("data_management.io.wp3.read_kp.KPReader._check_data_folder", return_value=None, autospec=True)
    def test_read_standard(self, mocker, mocker2):
        folder = "/not/existing/path/"
        sub_folder = "/not/existing/sub/folder/"
        reader = KPEnsembleReader(wp3_output_folder=folder, sub_folder=sub_folder)
        data, timestamp_data = reader.read(model_name="name_model", requested_date=self.DATE)
        assert timestamp_data == self.DATE
        assert isinstance(data, list)
        assert isinstance(timestamp_data, dt.datetime)
        for d in data:
            assert d.equals(self.KP)
