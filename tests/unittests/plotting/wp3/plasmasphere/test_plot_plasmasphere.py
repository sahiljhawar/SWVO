import sys
from os import path
import logging
import pytest

from datetime import datetime

import numpy as np
from scipy.stats import multivariate_normal


basepath = path.dirname(__file__)
sys.path.append(path.abspath(path.join(basepath, "../../../../../")))
from data_management.io.wp3.read_plasmasphere import PlasmaspherePredictionReader
from data_management.io.wp3.read_plasmasphere_combined_inputs import PlasmasphereCombinedInputsReader
from data_management.plotting.wp3.plasmasphere.plasmasphere_plot import PlasmaspherePlot


@pytest.fixture(scope="module")
def get_date():
    return datetime(year=2023, month=1, day=27, hour=13)


@pytest.fixture(scope="module")
def get_prediction_folder():
    return "../../../../data/data_management/plotting/wp3"


@pytest.fixture(scope="module")
def get_density_single_prediction(get_date, get_prediction_folder):

    plasmasphere_prediction_reader = \
        PlasmaspherePredictionReader(wp3_output_folder=get_prediction_folder)
    df_predictions = plasmasphere_prediction_reader.read(
        source="gfz_plasma",
        requested_date=get_date
    )
    return df_predictions


@pytest.fixture(scope="function")
def get_combined_inputs_reader(get_prediction_folder):
    return PlasmasphereCombinedInputsReader(get_prediction_folder)


@pytest.fixture(scope="function")
def get_kp_input_single_prediction(get_date, get_combined_inputs_reader):
    inputs_reader = get_combined_inputs_reader
    return inputs_reader.read(source="kp", requested_date=get_date)


@pytest.fixture(scope="function")
def get_solar_wind_input_single_prediction(get_date,
                                           get_combined_inputs_reader):
    inputs_reader = get_combined_inputs_reader
    return inputs_reader.read(source="solar_wind", requested_date=get_date)


def add_fake_column_from_original_column_to_inputs(
        df, time_column, origin_column_name, new_column_name, absolute_shift,
        prediction_date
):
    df[new_column_name] = df[origin_column_name]
    df = get_time_dependent_shift_factor(df, time_column, absolute_shift,
                                         prediction_date)
    df.loc[df[time_column] > prediction_date, new_column_name] = \
        df.loc[df[time_column] > prediction_date][
            [new_column_name, "shift_factor"]].apply(
            lambda x: x[new_column_name] + x["shift_factor"],
            axis=1
        )
    df.drop(columns="shift_factor", inplace=True)
    return df

def get_fake_value_density(l, mlt, original_value, shift_factor):
    return (original_value +
            16 * shift_factor * multivariate_normal.pdf(
                x=[l*np.cos(mlt*2*np.pi/24), l*np.sin(mlt*2*np.pi/24)],
                mean=[0, 0], cov=[2, 4]
            )
            )

def get_time_dependent_shift_factor(df, time_column, absolute_shift,
                                    prediction_date):

    df_shift_factor = df[[time_column]].drop_duplicates()
    df_shift_factor = df_shift_factor.assign(shift_factor=np.nan)
    df_shift_factor.loc[
        df[time_column] > prediction_date, "shift_factor"] = \
        df_shift_factor.loc[
            df[time_column] > prediction_date][time_column].apply(
            lambda x: absolute_shift * np.random.randint(-1, 2)
        )
    return df.merge(df_shift_factor, on=[time_column], how="left")

def add_fake_column_to_density(
        df, time_column, l_column, mlt_column, origin_column_name,
        new_column_name, absolute_shift, prediction_date
):
    df[new_column_name] = df[origin_column_name]

    df = get_time_dependent_shift_factor(df, time_column, absolute_shift,
                                         prediction_date)
    df.loc[df[time_column] > prediction_date, new_column_name] = \
        df.loc[df[time_column] > prediction_date][
            [l_column, mlt_column, origin_column_name, "shift_factor"]].apply(
            lambda x: get_fake_value_density(x[l_column], x[mlt_column],
                                             x[origin_column_name],
                                             x["shift_factor"]),
            axis=1
        )
    df.drop(columns="shift_factor", inplace=True)
    return df

@pytest.fixture(scope="function")
def get_density_ensemble_prediction(get_density_single_prediction, get_date):
    df_predictions_ensemble = get_density_single_prediction
    prediction_date = get_date
    for (name, absolute_shift) in zip(
            ["predicted_densities_1", "predicted_densities_2",
             "predicted_densities_3", "predicted_densities_4"],
            [1, 2.5, 1.5, 3]):
        df_predictions_ensemble = \
            add_fake_column_to_density(df_predictions_ensemble, "t", "L", "MLT",
                                       "predicted_densities", name,
                                       absolute_shift, prediction_date)

    # df_predictions_ensemble["predicted_densities_1"] = df_predictions_ensemble["predicted_densities"]
    # df_predictions_ensemble["predicted_densities_2"] = df_predictions_ensemble["predicted_densities"]
    # df_predictions_ensemble["predicted_densities_3"] = df_predictions_ensemble["predicted_densities"]
    # df_predictions_ensemble["predicted_densities_4"] = df_predictions_ensemble["predicted_densities"]

    return df_predictions_ensemble


@pytest.fixture(scope="function")
def get_kp_input_ensemble_prediction(get_kp_input_single_prediction, get_date):
    df_kp_ensemble = get_kp_input_single_prediction
    prediction_date = get_date
    for (name, absolute_shift) in zip(["kp_1", "kp_2", "kp_3", "kp_4"],
                                      [0.33333, 0.6666, 1.33333, 1.6666]):
        df_kp_ensemble = \
            add_fake_column_from_original_column_to_inputs(
                df_kp_ensemble, "t", "kp", name, absolute_shift,
                prediction_date
            )
    return df_kp_ensemble


@pytest.fixture(scope="function")
def get_solar_wind_input_ensemble_prediction(
        get_solar_wind_input_single_prediction, get_date
):

    df_solar_wind_ensemble = get_solar_wind_input_single_prediction
    prediction_date = get_date

    for (name, absolute_shift) in zip(["Bz_1", "Bz_2", "Bz_3", "Bz_4"],
                                      [1, 0.5, 1.7, 0.8]):
        df_solar_wind_ensemble = \
            add_fake_column_from_original_column_to_inputs(
                df_solar_wind_ensemble, "t", "Bz", name, absolute_shift,
                prediction_date
            )
    for (name, absolute_shift) in zip(["proton_density_1", "proton_density_2",
                                       "proton_density_3", "proton_density_4"], [6, 8, 4, 2]):
        df_solar_wind_ensemble = \
            add_fake_column_from_original_column_to_inputs(
                df_solar_wind_ensemble, "t", "proton_density", name,
                absolute_shift, prediction_date
            )
    for (name, absolute_shift) in zip(["speed_1", "speed_2", "speed_3", "speed_4"],
                                      [20, 30, 40, 60]):
        df_solar_wind_ensemble = \
            add_fake_column_from_original_column_to_inputs(
                df_solar_wind_ensemble, "t", "speed", name, absolute_shift,
                prediction_date
            )
    return df_solar_wind_ensemble


class TestPlasmaspherePlot:

    def test_plot_single_prediction(self,
                                    get_density_single_prediction,
                                    get_kp_input_single_prediction,
                                    get_solar_wind_input_single_prediction):

        df_predictions = get_density_single_prediction
        df_kp = get_kp_input_single_prediction
        df_solar_wind = get_solar_wind_input_single_prediction

        plasmasphere_plot = PlasmaspherePlot()
        plasmasphere_plot.plot_output(
            data=df_predictions,
            output_folder="./output/",
            video_file_name="plasma_video_single_prediction.mp4",
            kp_inputs=df_kp,
            solar_wind_inputs=df_solar_wind
        )

    def test_plot_ensemble_predictions(
            self,
            get_density_ensemble_prediction,
            get_kp_input_ensemble_prediction,
            get_solar_wind_input_ensemble_prediction
    ):
        df_predictions = get_density_ensemble_prediction
        df_kp = get_kp_input_ensemble_prediction
        df_solar_wind = get_solar_wind_input_ensemble_prediction

        plasmasphere_plot = PlasmaspherePlot()
        plasmasphere_plot.plot_output(
            data=df_predictions,
            output_folder="./output/",
            video_file_name="plasma_video_ensemble_prediction.mp4",
            kp_inputs=df_kp,
            solar_wind_inputs=df_solar_wind
        )

        assert 1 == 0
