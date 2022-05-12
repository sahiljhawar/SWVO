from data_management.io.base_file_reader import BaseReader


class TestBaseReader(object):

    def test_init(self):
        reader = BaseReader()
        assert reader.data_folder is None

    def test_read(self):
        reader = BaseReader()
        try:
            reader.read()
            assert False
        except NotImplementedError:
            assert True

    def test_check_data_folder(self):
        reader = BaseReader()
        try:
            reader._check_data_folder()
            assert False
        except NotImplementedError:
            assert True
