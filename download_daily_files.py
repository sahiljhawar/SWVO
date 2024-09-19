import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
import traceback

from data_management.io.kp import KpNiemegk, KpSWPC
from data_management.io.hp import Hp30GFZ, Hp60GFZ
from data_management.io.omni import OMNILowRes

base_path = '/home/bhaas/FLAG_TEST/'
kp_path = base_path + 'Kp/'

ENV_VAR_NAMES = {'DAILY_DOWNLOAD_LOG_DIR': base_path + 'log/',
                 'RT_KP_NIEMEGK_STREAM_DIR': kp_path + 'Niemegk/',
                 'RT_KP_SWPC_STREAM_DIR': kp_path + 'SWPC/',
                 'OMNI_LOW_RES_STREAM_DIR': base_path + 'OMNI/Low_res',
                 'RT_HP_GFZ_STREAM_DIR' : base_path + 'HP/'}


if __name__ == '__main__':

    for var in ENV_VAR_NAMES.keys():
        os.environ[var] = ENV_VAR_NAMES[var]

    Path(os.environ.get('DAILY_DOWNLOAD_LOG_DIR')).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=os.environ.get('DAILY_DOWNLOAD_LOG_DIR') + f'daily_downloads_log_{datetime.today().strftime("%Y%m%d")}.log',
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
    
    date_yesterday_start = datetime.today().replace(hour=0, minute=0, second=1) - timedelta(days=1)
    date_yesterday_end = date_yesterday_start.replace(hour=23, minute=59, second=59) 

    logging.info(f'Target time from {date_yesterday_start} to {date_yesterday_end}')

    logging.info('Starting downloading and processing...')

    logging.info('Kp Niemegk...\n')
    try:
        KpNiemegk().download_and_process(date_yesterday_start, datetime.today(), reprocess_files=True)
    except:
        logging.error('Encountered error while downloading Niemegk Kp. Traceback:')
        logging.error(traceback.format_exc())

    logging.info('Kp SWPC...\n')
    try:
        KpSWPC().download_and_process(datetime.today(), reprocess_files=True)
    except:
        logging.error('Encountered error while downloading SWPC Kp. Traceback:')
        logging.error(traceback.format_exc())

    logging.info('OMNI low resolution...\n')
    try:
        OMNILowRes().download_and_process(date_yesterday_start, datetime.today(), reprocess_files=True)
    except:
        logging.error('Encountered error while downloading OMNI low res. Traceback:')
        logging.error(traceback.format_exc())

    logging.info('Finished downloading and processing!')

    logging.info('Hp30 GFZ...\n')
    try:
        Hp30GFZ().download_and_process(date_yesterday_start, datetime.today(), reprocess_files=True)
    except:
        logging.error('Encountered error while downloading Hp30. Traceback:')
        logging.error(traceback.format_exc())

    logging.info('Hp60 GFZ...\n')
    try:
        Hp60GFZ().download_and_process(date_yesterday_start, datetime.today(), reprocess_files=True)
    except:
        logging.error('Encountered error while downloading Hp30. Traceback:')
        logging.error(traceback.format_exc())

    logging.info('Finished downloading and processing!')