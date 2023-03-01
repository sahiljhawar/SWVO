import sys
import os

import pytest

import numpy as np
import pandas as pd

from datetime import datetime, timedelta
import matplotlib.pyplot as plt


basepath = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basepath, "../../../../")))
from data_management.plotting.wp3.hp60.plot_hp60 import PlotHp60


def generate_fake_kp():
    kp_values = [0, 0.333, 0.666, 1, 1.333, 1.666, 2, 2.333, 2.666,
                 3, 3.333, 3.666, 4, 4.333, 4.666, 5, 5.333, 5.666,
                 6, 6.333, 6.666, 7, 7.333, 7.666, 8]
    return np.random.choice(kp_values)


@pytest.fixture(scope="function")
def build_fake_hp60_data():

    start_date = datetime(year=2023, month=3, day=1)
    dates = pd.date_range(start_date, start_date + timedelta(days=3),
                          freq="1H")
    df = pd.DataFrame(index=dates)
    df.index.name = "t"
    df["Hp60"] = np.nan
    df["Hp60"] = df["Hp60"].apply(lambda x: generate_fake_kp())
    return df


class TestPlotHp60:

    def test_plot_output(self, build_fake_hp60_data):

        df = build_fake_hp60_data

        print(df)

        PlotHp60.plot_output(df)
        plt.savefig(os.path.abspath(os.path.join(basepath,
                                                 "./output/hp60_fake.png")))

        assert 1 == 0
