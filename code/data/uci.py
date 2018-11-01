import numpy as np

from data import DataLoader, Config
import os


class UCI(DataLoader):

    def __init__(self, conf:Config):
        super().__init__(conf)

    def generate_data(self):
        file = self.conf.params["file"]
        delimiter = self.conf.params["delimiter"]
        uci_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'uci')
        data = np.loadtxt(os.path.join(uci_dir,file), skiprows=1, delimiter=delimiter)

        return data






