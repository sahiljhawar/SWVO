from unittest import mock
import os
import inspect
import sys
import pandas as pd
import numpy as np

LOCAL_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(LOCAL_PATH, "../../../../"))
from data_management.plotting.wp3.kp.plot_kp import PlotKpOutput


class TestPlotKP(object):

    def test_init_folder_found(self):
        PlotKpOutput()
        assert True
