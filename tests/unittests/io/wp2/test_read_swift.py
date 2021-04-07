from unittest import mock
from data_management.io.wp2.read_swift import SwiftReader


class TestReadSWIFT(object):

    @mock.patch("data_management.io.wp2.read_swift._read_single_file", return_value=SWEPAM_TEST_1, autospec=True)
    def test_read_standard_use(self):
        reader = SwiftReader()
        data_gsm, data_hgc = reader.read(date=None, fields=None)
