import os
from pathlib import Path
from typing import Tuple
from datetime import datetime, timedelta

import pandas as pd

class HpEnsemble(object):

    ENV_VAR_NAME = 'PLACEHOLDER; SEE DERIVED CLASSES BELOW'

    def __init__(self, index, data_dir:str|Path=None):
        if data_dir is None:

            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f'Necessary environment variable {self.ENV_VAR_NAME} not set!')

            data_dir = os.environ.get(self.ENV_VAR_NAME)

        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(f'Data directory {self.data_dir} does not exist! Impossible to retrive data!')

        self.index = index
        if self.index not in ["hp30", "hp60"]:
            raise RuntimeError(f"Requested {self.index} index does not exist! Possible options: hp30, hp60")
        
        self.index_number = index[2:]

    def read(self, start_time:datetime, end_time:datetime) -> list:

        if start_time is None:
            start_time = datetime.utcnow().replace(microsecond=0, minute=0, second=0)

        if end_time is None:
            end_time = start_time + timedelta(days=3)

        start_date = datetime(start_time.year, start_time.month, start_time.day)
        str_date = start_date.strftime("%Y%m%dT%H%M%S")

        file_list = sorted(self.data_dir.glob(f"FORECAST_HP30_SWIFT_DRIVEN_swift_{str_date}_ensemble_*.csv"))

        data = []
        for file in file_list:
            df = pd.read_csv(file, names=["t", self.index])

            df["t"] = pd.to_datetime(df["t"])
            df.index = df["t"]
            df.drop(labels=["t"], axis=1, inplace=True)
            df = df.truncate(before=start_time-timedelta(minutes=int(self.index_number)-0.01), after=end_time+timedelta(minutes=int(self.index_number)+0.01))

            data.append(df)
            
        return data
    
class Hp30Ensemble(HpEnsemble):

    ENV_VAR_NAME = 'HP30_ENSEMBLE_FORECAST_DIR'

    def __init__(self, data_dir:str|Path=None):
        super().__init__('hp30', data_dir)

class Hp60Ensemble(HpEnsemble):

    ENV_VAR_NAME = 'HP60_ENSEMBLE_FORECAST_DIR'

    def __init__(self, data_dir:str|Path=None):
        super().__init__('hp60', data_dir)