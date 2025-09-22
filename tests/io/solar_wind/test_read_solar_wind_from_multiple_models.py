# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from swvo.io.exceptions import ModelError
from swvo.io.solar_wind import (
    DSCOVR,
    SWACE,
    SWOMNI,
    SWSWIFTEnsemble,
    read_solar_wind_from_multiple_models,
)

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = Path(os.path.join(TEST_DIR, "data/"))


class TestReadSolarWindFromMultipleModels:
    @pytest.fixture(scope="session", autouse=True)
    def set_env_var(self):
        ENV_VAR_NAMES = {
            "OMNI_HIGH_RES_STREAM_DIR": f"{str(DATA_DIR)}",
            "SWIFT_ENSEMBLE_OUTPUT_DIR": f"{str(DATA_DIR)}/ensemble",
            "RT_SW_ACE_STREAM_DIR": f"{str(DATA_DIR)}/ACE_RT",
            "SW_DSCOVR_STREAM_DIR": f"{str(DATA_DIR)}/DSCOVR",
        }

        for key, var in ENV_VAR_NAMES.items():
            os.environ[key] = ENV_VAR_NAMES[key]

    @pytest.fixture
    def sample_times(self):
        now = datetime(2024, 11, 25).replace(tzinfo=timezone.utc, minute=0, second=0, microsecond=0)
        return {
            "past_start": now - timedelta(days=5),
            "past_end": now - timedelta(days=2),
            "future_start": now + timedelta(days=1),
            "future_end": now + timedelta(days=3),
            "test_time_now": now,
        }

    @pytest.fixture
    def expected_columns(self):
        return [
            "proton_density",
            "speed",
            "bavg",
            "temperature",
            "bx_gsm",
            "by_gsm",
            "bz_gsm",
            "file_name",
        ]

    def test_basic_historical_read(self, sample_times, expected_columns):
        data = read_solar_wind_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["past_end"],
            model_order=[SWOMNI(), DSCOVR(), SWACE()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        assert isinstance(data, pd.DataFrame)
        assert all(col in data.columns for col in expected_columns)
        assert data.loc["2024-11-22 18:00:00+00:00"].model == "omni"
        assert data.loc["2024-11-23 00:00:00+00:00"].model == "dscovr"
        # no ace since dscovr and ace files are same in the test data and dscovr is before ace in the model_order
        assert not data["file_name"].isna().all()

    def test_basic_forecast_read(self, sample_times, expected_columns):
        data = read_solar_wind_from_multiple_models(
            start_time=sample_times["future_start"],
            end_time=sample_times["future_end"],
            model_order=[SWSWIFTEnsemble()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )
        assert all(isinstance(d, pd.DataFrame) for d in data)
        assert all(not d["file_name"].isna().all() for d in data)

        assert all(all(d.model == "swift") for d in data)
        for d in data:
            assert all(col in d.columns for col in expected_columns)

    def test_full_ensemble(self, sample_times, expected_columns):
        data = read_solar_wind_from_multiple_models(
            start_time=sample_times["future_start"],
            end_time=sample_times["future_end"],
            model_order=[SWSWIFTEnsemble()],
            reduce_ensemble=None,
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        assert isinstance(data, list)
        assert len(data) > 1
        assert all(isinstance(d, pd.DataFrame) for d in data)
        assert all(not d["file_name"].isna().all() for d in data)
        for d in data:
            assert all(col in d.columns for col in expected_columns)

    def test_time_ordering_and_transition(self, sample_times, expected_columns):
        data = read_solar_wind_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["future_end"],
            model_order=[SWOMNI(), DSCOVR(), SWACE(), SWSWIFTEnsemble()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        for d in data:
            assert d.index.is_monotonic_increasing
            assert d.loc["2024-11-20 00:00:00+00:00"].model == "omni"
            assert d.loc["2024-11-25 00:00:00+00:00"].model == "dscovr"
            assert d.loc["2024-11-25 00:01:00+00:00"].model == "swift"
            assert all(col in d.columns for col in expected_columns)

    def test_forecast_in_past(self, sample_times, expected_columns):
        data = read_solar_wind_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["past_end"],
            model_order=[SWOMNI(), DSCOVR(), SWACE(), SWSWIFTEnsemble()],
            historical_data_cutoff_time=sample_times["test_time_now"] - timedelta(days=3),
        )

        for d in data:
            assert d.index.is_monotonic_increasing
            assert d.loc["2024-11-22 00:01:00+00:00"].model == "swift"
            assert d.loc["2024-11-22 00:00:00+00:00"].model == "omni"
            assert all(col in d.columns for col in expected_columns)

    def test_time_boundaries(self, sample_times):
        start = sample_times["past_start"]
        end = sample_times["future_end"]

        data = read_solar_wind_from_multiple_models(
            start_time=start,
            end_time=end,
            model_order=[SWOMNI(), SWSWIFTEnsemble()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        for d in data:
            assert d.index.min() >= start
            assert d.index.max() <= end

    def test_invalid_time_range(self, sample_times):
        with pytest.raises(AssertionError):
            read_solar_wind_from_multiple_models(
                start_time=sample_times["future_end"],
                end_time=sample_times["past_start"],
                model_order=[SWOMNI()],
            )

    def test_data_consistency(self, sample_times):
        params = {
            "start_time": sample_times["past_start"],
            "end_time": sample_times["future_start"],
            "model_order": [SWOMNI(), DSCOVR(), SWACE(), SWSWIFTEnsemble()],
            "historical_data_cutoff_time": sample_times["test_time_now"],
        }

        data1 = read_solar_wind_from_multiple_models(**params)
        data2 = read_solar_wind_from_multiple_models(**params)

        for d1, d2 in zip(data1, data2):
            pd.testing.assert_frame_equal(d1, d2)

    def test_synthetic_now_time_deprecation_with_message(self, sample_times):
        with pytest.warns(DeprecationWarning, match="synthetic_now_time.*deprecated"):
            read_solar_wind_from_multiple_models(
                start_time=sample_times["past_start"],
                end_time=sample_times["future_end"],
                synthetic_now_time=sample_times["test_time_now"],
            )

    def test_model_check_with_wrong_class(self, sample_times):
        class FakeModel:
            pass

        fake = FakeModel()
        with pytest.raises(ModelError, match="Unknown or incompatible model"):
            read_solar_wind_from_multiple_models(
                start_time=sample_times["past_start"],
                end_time=sample_times["future_end"],
                model_order=[fake],
            )

    def test_27_day_recurrence_basic(self, sample_times, expected_columns):
        data = read_solar_wind_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["past_end"],
            model_order=[SWOMNI(), DSCOVR(), SWACE()],
            historical_data_cutoff_time=sample_times["test_time_now"],
            recurrence=True,
        )

        assert isinstance(data, pd.DataFrame)
        assert all(col in data.columns for col in expected_columns)

        recurrence_models = data[data["model"].str.contains("recurrence", na=False)]
        if not recurrence_models.empty:
            assert any("_recurrence_27d" in model for model in recurrence_models["model"].unique())
        assert data.index.is_monotonic_increasing
        assert data.index.freq == "1min"

    def test_3_hour_interpolation_with_recurrence(self, sample_times, expected_columns):
        # Use a longer time range to increase chances of gaps that need interpolation
        extended_start = sample_times["past_start"] - timedelta(days=2)
        extended_end = sample_times["past_end"] + timedelta(days=1)

        data_no_rec = read_solar_wind_from_multiple_models(
            start_time=extended_start,
            end_time=extended_end,
            model_order=[SWOMNI(), DSCOVR(), SWACE()],
            historical_data_cutoff_time=sample_times["test_time_now"],
            recurrence=False,
            download=False,
        )

        data_with_rec = read_solar_wind_from_multiple_models(
            start_time=extended_start,
            end_time=extended_end,
            model_order=[SWOMNI(), DSCOVR(), SWACE()],
            historical_data_cutoff_time=sample_times["test_time_now"],
            recurrence=True,
            download=True,
        )

        nan_count_no_rec = data_no_rec.isna().sum()
        nan_count_with_rec = data_with_rec.isna().sum()

        for i in expected_columns:
            assert nan_count_with_rec[i] <= nan_count_no_rec[i]
