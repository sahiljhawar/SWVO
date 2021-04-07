import sys
import datetime as dt
import os
import inspect

LOCAL_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(LOCAL_PATH, "../../../../"))
from data_management.io.wp2.read_swift import SwiftReader


class TestReadSWIFT(object):

    def test_read_standard_use(self):
        reader = SwiftReader()
        data_gsm, data_hgc = reader.read(date=dt.datetime(2021, 4, 7), fields=None)
        print(data_hgc)
