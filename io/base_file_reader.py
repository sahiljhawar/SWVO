from abc import abstractmethod


class BaseReader(object):
    def __init__(self):
        pass

    @abstractmethod
    def read(self, *args):
        raise NotImplementedError
