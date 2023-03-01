import sys
import os

import pytest

from datetime import datetime

basepath = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basepath, "../../../../")))
from data_management.io.wp3.read_hp60 import Hp60Reader


@pytest.fixture(scope="function")
def build_hp60_reader():
    return Hp60Reader(hp60_output_folder=os.path.abspath(
        os.path.join(basepath, "../../../data/io/wp3/hp60/")
    )
    )


class TestHp60Reader:

    def test_read_files_generated_before_2023(self,
                                              build_hp60_reader):
        hp60_reader = build_hp60_reader
        hp60, _ = hp60_reader.read(requested_date=datetime(year=2022, month=4,
                                                           day=17, hour=14),
                                   header=True)
        assert hp60 is None

    def test_read_files_generated_from_2023(self,
                                            build_hp60_reader):
        hp60_reader = build_hp60_reader
        hp60, _ = hp60_reader.read(requested_date=datetime(year=2023, month=2,
                                                           day=24, hour=16))
        assert hp60 is None
