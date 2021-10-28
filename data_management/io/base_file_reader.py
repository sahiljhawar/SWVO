from abc import abstractmethod


class BaseReader(object):
    def __init__(self):
        pass

    @abstractmethod
    def read(self, *args):
        raise NotImplementedError

    @abstractmethod
    def _check_data_folder(self):
        raise NotImplementedError
