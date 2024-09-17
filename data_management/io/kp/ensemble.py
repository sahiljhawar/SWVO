import os
from pathlib import Path
from typing import Tuple
from datetime import datetime, timedelta

import pandas as pd

class KpEnsemble(object):

    ENV_VAR_NAME = 'KP_ENSEMBLE_FORECAST_DIR'

    def __init__(self, data_dir:str|Path=None):
        if data_dir is None:

            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f'Necessary environment variable {self.ENV_VAR_NAME} not set!')

            data_dir = os.environ.get(self.ENV_VAR_NAME)

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def read(self, start_time:datetime, end_time:datetime) -> list:

        if start_time is None:
            start_time = datetime.utcnow().replace(microsecond=0, minute=0, second=0)

        if end_time is None:
            end_time = start_time + timedelta(days=3)

        start_date = datetime(start_time.year, start_time.month, start_time.day)
        str_date = start_date.strftime("%Y%m%dT%H%M%S")

        file_list = sorted(self.data_dir.glob(f"FORECAST_PAGER_SWIFT_swift_{str_date}_ensemble_*.csv"))

        data = []
        for file in file_list:
            df = pd.read_csv(file, names=["t", "kp"])

            df["t"] = pd.to_datetime(df["t"])
            df.index = df["t"]
            df.drop(labels=["t"], axis=1, inplace=True)
            df = df.truncate(before=start_time-timedelta(hours=2.9999), after=end_time+timedelta(hours=2.9999))

            data.append(df)
            
        return data