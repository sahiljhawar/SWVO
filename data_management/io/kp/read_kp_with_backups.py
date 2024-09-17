from datetime import datetime

import pandas as pd
import numpy as np

from data_management.io.kp import KpOMNI, KpNiemegk, KpSWPC, KpEnsemble

def read_kp_with_backups(start_time:datetime, end_time:datetime, model_order:list=None, reduce_ensemble=None):
    
    if model_order is None:
        model_order = [KpOMNI(), KpNiemegk(), KpEnsemble(), KpSWPC()]

    data_out = [pd.DataFrame()]

    for model in model_order:

        if isinstance(model, KpOMNI):
            print('Reading omni...')
            data_one_model = [model.read(start_time, end_time, download=True)]
            model_label = 'omni'

        if isinstance(model, KpNiemegk):
            print('Reading niemegk...')
            data_one_model = [model.read(start_time, end_time, download=True)]
            model_label = 'niemegk'

        # Forecasting models are called with today's date
        if isinstance(model, KpSWPC):
            print('Reading swpc...')
            data_one_model = [model.read(datetime.today().replace(hour=0, minute=0, second=0), end_time, download=True)]
            model_label = 'swpc'

        if isinstance(model, KpEnsemble):
            print('Reading PAGER Kp ensemble...')
            data_one_model = model.read(datetime.today(), end_time)
            model_label = 'ensemble'

            num_ens_members = len(data_one_model)

            if reduce_ensemble == 'mean':

                kp_mean_ensembles = []

                for it in range(len(data_one_model[0].index)):
                    data_curr_time = []
                    for ie in range(num_ens_members):
                        data_curr_time.append(data_one_model[ie].loc[data_one_model[ie].index[it], 'kp'])

                    kp_mean_ensembles.append(np.mean(data_curr_time))

                data_one_model = [pd.DataFrame(index=data_one_model[0].index, data={'kp': kp_mean_ensembles})]

            elif reduce_ensemble is None:
                data_out = data_out * num_ens_members

        any_nans_found = False
        # we making it a list in case of ensemble memblers
        for i in range(len(data_one_model)):
            data_one_model[i]['model'] = model_label
            data_one_model[i].loc[data_one_model[i]['kp'].isna(), 'model'] = None
            data_out[i] = data_out[i].combine_first(data_one_model[i])

            if data_out[i]['kp'].isna().sum() > 0:
                any_nans_found = True

        print(data_out)

        # if no NaNs are present anymore, we don't have to read backups
        if not any_nans_found:
            break

    if len(data_out) == 1:
        data_out = data_out[0]

    return data_out