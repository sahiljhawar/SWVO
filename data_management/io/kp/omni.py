import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from data_management.io.omni import OMNILowRes

class KpOMNI(object):

    ENV_VAR_NAME = 'OMNI_LOW_RES_STREAM_DIR'

    START_YEAR = 1963

    def __init__(self, data_dir=None):
        
        if data_dir is None:

            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f'Necessary environment variable {self.ENV_VAR_NAME} not set!')

            data_dir = os.environ.get(self.ENV_VAR_NAME)

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.omni_low_res = OMNILowRes(data_dir)

    def read(self, start_time:datetime, end_time:datetime, download:bool=False) -> pd.DataFrame:

        if start_time < datetime(self.START_YEAR, 1, 1):
            print("Start date chosen falls behind the existing data. Moving start date to first"
                  " available mission files...")
            start_time = datetime(self.START_YEAR, 1, 1)

        file_paths, _ = self.omni_low_res._get_processed_file_list(start_time, end_time)

        dfs = []

        for file_path in file_paths:

            if not file_path.exists():
                if download:
                    self.omni_low_res.download_and_process(start_time, end_time)
                else:
                    print(f'File {file_path} not found, filling with NaNs')
                    raise NotImplementedError()
            
            dfs.append(self._read_single_file(file_path))

        data_out = pd.concat(dfs)

        # we return it just every 3 hours
        data_out.drop(data_out[data_out.index.hour % 3 != 0].index, axis=0, inplace=True)
        data_out = data_out.truncate(before=start_time-timedelta(hours=2.9999), after=end_time+timedelta(hours=2.9999))

        return data_out

    def _read_single_file(self, file_path) -> pd.DataFrame:

        df = pd.read_csv(file_path)

        df["t"] = pd.to_datetime(df["timestamp"])
        df.index = df["t"]
        df.drop(labels=["t"], axis=1, inplace=True)
        df.drop(labels=["timestamp"], axis=1, inplace=True)
        df.drop(labels=["dst"], axis=1, inplace=True)
        
        df["file_name"] = file_path
        df.loc[df["kp"].isna(), "file_name"] = None

        return df
