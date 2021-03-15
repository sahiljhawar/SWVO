from abc import abstractmethod


class PlotOutput(object):
    def __init__(self):
        self.description = None

    @staticmethod
    @abstractmethod
    def plot_output(data):
        raise NotImplementedError
