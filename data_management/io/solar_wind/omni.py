import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from data_management.io.omni import OMNIHighRes

class SWOMNI(object):

    ENV_VAR_NAME = 'OMNI_HIGH_RES_STREAM_DIR'

    START_YEAR = 1981

    def __init__(self, data_dir=None):
        
        if data_dir is None:

            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f'Necessary environment variable {self.ENV_VAR_NAME} not set!')

            data_dir = os.environ.get(self.ENV_VAR_NAME)

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.omni_high_res = OMNIHighRes(data_dir)

    def read(self, start_time:datetime, end_time:datetime, cadence_min:float=1, download:bool=False) -> pd.DataFrame:

        if start_time < datetime(self.START_YEAR, 1, 1):
            print("Start date chosen falls behind the existing data. Moving start date to first"
                  " available mission files...")
            start_time = datetime(self.START_YEAR, 1, 1)

        assert start_time < end_time

        file_paths, _ = self.omni_high_res._get_processed_file_list(start_time, end_time, cadence_min)

        dfs = []

        for file_path in file_paths:

            if not file_path.exists():
                if download:
                    self.omni_high_res.download_and_process(start_time, end_time, cadence_min=cadence_min)
                else:
                    print(f'File {file_path} not found, filling with NaNs')
                    raise NotImplementedError()
            
            dfs.append(self._read_single_file(file_path))

        data_out = pd.concat(dfs)
        data_out = data_out.truncate(before=start_time-timedelta(minutes=cadence_min-0.0000001), after=end_time+timedelta(minutes=cadence_min+0.0000001))

        return data_out

    def _read_single_file(self, file_path) -> pd.DataFrame:

        df = pd.read_csv(file_path)

        df["t"] = pd.to_datetime(df["timestamp"])
        df.index = df["t"]
        df.drop(labels=["t"], axis=1, inplace=True)
        df.drop(labels=["timestamp"], axis=1, inplace=True)
        
        nan_mask = df.isna().all(axis=1)
        df["file_name"] = file_path
        df.loc[nan_mask, "file_name"] = None

        return df
