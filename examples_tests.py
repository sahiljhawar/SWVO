from datetime import datetime, timedelta
import os

from data_management.io.kp import KpNiemegk, KpSWPC, KpOMNI
from data_management.io.omni import OMNILowRes
from data_management.io.kp import read_kp_with_backups
from data_management.io.hp import Hp30GFZ, Hp60GFZ, Hp30Ensemble

from matplotlib import pyplot as plt

os.environ['RT_KP_NIEMEGK_STREAM_DIR'] = '/home/bhaas/FLAG_TEST/Niemegk/'
os.environ['RT_KP_SWPC_STREAM_DIR'] = '/home/bhaas/FLAG_TEST/SWPC/'
os.environ['OMNI_LOW_RES_STREAM_DIR'] = '/home/bhaas/FLAG_TEST/OMNI_LOW_RES/'
os.environ['KP_ENSEMBLE_FORECAST_DIR'] = '/PAGER/WP3/data/outputs/SWIFT_ENSEMBLE/'
os.environ['RT_HP_GFZ_STREAM_DIR'] = '/home/bhaas/FLAG_TEST/Hp/'
os.environ['HP30_ENSEMBLE_FORECAST_DIR'] = '/PAGER/WP3/data/outputs/HP30/'

start_time = datetime.today() - timedelta(days=15)
end_time = datetime.today() - timedelta(days=2)

start_time = datetime(start_time.year, start_time.month, start_time.day) + timedelta(seconds=1)
end_time = datetime(end_time.year, end_time.month, end_time.day) - timedelta(seconds=1)

print(start_time)
print(end_time)

#KpNiemegk().download_and_process(datetime.today()-timedelta(days=5), datetime.today()+timedelta(days=2), reprocess_files=True, verbose=True)
#print(KpNiemegk().read(start_time, end_time))

#KpSWPC().download_and_process(datetime.today(), reprocess_files=True, verbose=True)
#print(KpSWPC().read(datetime(2024, 5, 10)))

#OMNILowRes().download_and_process(start_time, end_time, reprocess_files=True, verbose=True)
#print(KpOMNI().read(start_time, end_time))

#Hp60GFZ().download_and_process(datetime.today(), datetime.today(), reprocess_files=True, verbose=True)
#print(Hp60GFZ().read(datetime.today()-timedelta(days=365), datetime.today()+timedelta(days=10)))

print(Hp30Ensemble().read(datetime.today(), datetime.today()+timedelta(hours=30)))

# data_kp_all = [read_kp_with_backups(datetime.today()-timedelta(days=12), datetime.today()-timedelta(days=5))]

# data_omni = data_kp_all[0].loc[data_kp_all[0]['model'] == 'omni']
# data_niemegk = data_kp_all[0].loc[data_kp_all[0]['model'] == 'niemegk']

# plt.subplots(1,1,figsize=(19/2,10/2))
# plt.step(data_omni.index, data_omni['kp'], 'b')
# plt.step(data_niemegk.index, data_niemegk['kp'], 'r')

# for ie in range(len(data_kp_all)):
#      data_ensemble = data_kp_all[ie].loc[data_kp_all[ie]['model'] == 'ensemble']
#      plt.step(data_ensemble.index, data_ensemble['kp'], 'g')

# plt.legend(["Omni", "Niemegk", "Ensemble"])
# plt.grid()
# plt.savefig('test.png')