from abc import abstractmethod


class BasePlot(object):
    def __init__(self):
        pass

    @abstractmethod
    def plot(self, *args):
        raise NotImplementedError