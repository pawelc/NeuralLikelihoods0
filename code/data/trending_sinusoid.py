import numpy as np

from data import DataLoader, Config


class TrendingSinusoid(DataLoader):

    def __init__(self, conf:Config):
        super().__init__(conf)

    def generate_data(self):
        samples = 10000
        noise = "normal"
        if "noise" in self.conf.params:
            noise = self.conf.params["noise"]
        y_data = np.float32(np.random.uniform(-20.5, 20.5, samples))
        if noise == "normal":
            r_data = np.float32(np.random.normal(size=samples, scale=1))
        elif noise == "standard_t":
            r_data = np.float32(np.random.standard_t(self.conf.params["df"], samples))

        x_data = np.float32(np.sin(0.4 * y_data) * 7.0 + y_data * 0.5)

        y_data = (y_data+r_data * 1.0).reshape(-1, 1)
        x_data = x_data.reshape(-1, 1)

        return np.c_[x_data, y_data]






