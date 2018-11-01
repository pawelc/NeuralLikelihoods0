from collections import OrderedDict
import random

import numpy as np
import pylab
from bokeh.plotting import figure, output_notebook, show
import os
import itertools
from bokeh.palettes import Category10
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd

from flags import FLAGS


class Config:
    def __init__(self, dir='data',x_slice=slice(None,-1), y_slice=slice(-1,None), normalize=False, uniquenessThreshold=None):
        self.dir = dir
        self.normalize = normalize
        self.x_slice = x_slice
        self.y_slice = y_slice
        self.params = OrderedDict()
        self.uniquenessThreshold=uniquenessThreshold

    def add_param(self, name, value):
        self.params[name]=value
        return self

    def id(self):
        id = ""
        for param_name,param_val in self.params.items():
            if type(param_val) is list:
                id += '_' + param_name+"_" + ("_".join([str(el) for el in param_val]))
            elif isinstance(param_val, np.ndarray):
                pass
            else:
                id += '_' + param_name + '_' + str(param_val)

        return id


class DataLoader:

    def __init__(self ,conf:Config):
        self.conf = conf
        os.makedirs(conf.dir, exist_ok=True)
        self.names = ["train", "validation", "test"]
        self.data, self.train_x, self.train_y, self.test_x, self.test_y, self.validation_x, self.validation_y = \
            None, None, None, None, None, None, None
        self.random_seed = None

    def generate_data(self):
        raise NotImplemented

    def can_compute_ll(self):
        return False

    def ll(self, x_data, y_data):
        raise NotImplemented

    def load_data(self):
        try:
            self.load_from_file()
            print("loaded data: %s"%self._file_name())
        except IOError:
            self.data = self.generate_data().astype(np.float32)
            if self.conf.uniquenessThreshold is not None:
                cat_columns = []
                for i in range(self.data.shape[1]):
                    uniqness = len(np.unique(self.data[:, i])) / len(self.data[:, i])
                    if uniqness < self.conf.uniquenessThreshold:
                        cat_columns.append(i)
                non_cat_columns = [i for i in range(self.data.shape[1]) if i not in cat_columns]
                self.data = self.data[:, non_cat_columns]

            self.sample_cross_validation(101)
            self.save_to_file()
            print("generated and saved data: %s" % self._file_name())

        self.min_x = np.min(self.train_x, axis=0)
        self.max_x = np.max(self.train_x, axis=0)
        self.min_y = np.min(self.train_y, axis=0)
        self.max_y = np.max(self.train_y, axis=0)

    def sample_cross_validation(self, random_seed):
        self.random_seed = random_seed
        train_data, test_data = train_test_split(self.data, test_size=0.2, random_state=random_seed)

        self.train_x = train_data[:,self.conf.x_slice]
        self.test_x = test_data[:,self.conf.x_slice]
        self.train_y = train_data[:,self.conf.y_slice]
        self.test_y = test_data[:,self.conf.y_slice]

        self.train_x, self.validation_x, self.train_y, self.validation_y = train_test_split(self.train_x, self.train_y,
                                                                                            test_size=0.2,
                                                                                            random_state=random_seed)
        if self.conf.normalize:
            mean_x = np.mean(self.train_x, axis=0, keepdims=True)
            std_x = np.std(self.train_x, axis=0, keepdims=True)
            mean_y = np.mean(self.train_y, axis=0, keepdims=True)
            std_y = np.std(self.train_y, axis=0, keepdims=True)

            self.train_x = (self.train_x-mean_x)/std_x
            self.train_y = (self.train_y-mean_y)/std_y

            self.test_x = (self.test_x - mean_x) / std_x
            self.test_y = (self.test_y - mean_y) / std_y

            self.validation_x = (self.validation_x - mean_x) / std_x
            self.validation_y = (self.validation_y - mean_y) / std_y

    def load_from_file(self):
        loaded = np.load(os.path.join(self.conf.dir, self._file_name()))
        self.data, self.train_x, self.train_y, self.test_x, self.test_y, self.validation_x, self.validation_y = \
            loaded['data'],loaded['train_x'],loaded['train_y'],loaded['test_x'],loaded['test_y'],loaded['validation_x'],loaded['validation_y']

    def _file_name(self):
        return "%s.npz" % (FLAGS.data_set)

    def save_to_file(self):
        np.savez(os.path.join(self.conf.dir, self._file_name()),
                 data=self.data,train_x=self.train_x, train_y=self.train_y,
                 test_x=self.test_x, test_y=self.test_y,
                 validation_x=self.validation_x, validation_y=self.validation_y)

    def plot_data(self, show=True):
        if self.train_x.shape[1] == 1 and self.train_y.shape[1] == 1:
            # 1-D data
            plt.figure(figsize=(16, 8));
            plt.title("Experimental  Data")
            plt.plot(self.train_x, self.train_y, 'ro', alpha=0.3 ,color="blue", label="train");
            plt.plot(self.test_x, self.test_y, 'ro', alpha=0.3, color="red", label="test");
            plt.plot(self.validation_x, self.validation_y, 'ro', alpha=0.3, color="green", label="validation");
            plt.legend()
            if show:
                plt.show();
        else:
            df = pd.DataFrame(np.c_[self.train_x, self.train_y], columns=['x%d'%i for i in range(self.train_x.shape[1])] +
                                                                         ['y%d' % i for i in
                                                                          range(self.train_y.shape[1])])
            fig, ax = plt.subplots(1,1,figsize=(16, 8));
            axes = pd.plotting.scatter_matrix(df, alpha=0.2, ax=ax, color="black", label="train", diagonal='kde', density_kwds={'color':'black'})

            # plt.tight_layout()
            if show:
                plt.show();
            return axes

def prepare_display():
    output_notebook()
    
def show_xy_data(x_data, y_data):
    p = figure(title="data", x_axis_label='x', y_axis_label='y')
    # add a line renderer with legend and line thickness
    p.scatter(x_data.flatten(), y_data.flatten(), legend="data", alpha=0.5)
    show(p);


def color_gen():
    yield from itertools.cycle(Category10[10])


def plot(y_data,y_data_grid, cdf_data_grid, cdf_est, pdf_est_nn, title):
    # dy = y_data_grid[1, 0] - y_data_grid[0, 0]
    # pdf_est_num = np.diff(cdf_data_grid.flatten()) / dy

    fig, ax1 = plt.subplots()
    plt.title(title)
    # ax1.plot(y_data_grid[1:, :], pdf_est_num, '.', label="pdf_est_num")
    ax1.plot(y_data, pdf_est_nn, label="pdf_est_nn")
    ax1.set_xlabel('y')
    ax1.set_ylabel('pdf0', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    sns.distplot(y_data, ax=ax1, color='b', hist=False, kde_kws={"color": "b", "label": "pdf_data", 'ls': '--'});

    ax2.plot(y_data, cdf_est.flatten(), 'r', label="cdf_est_nn")
    ax2.plot(y_data_grid, cdf_data_grid, '--r', label="cdf_data")
    ax2.set_ylabel('cdf0', color='r')
    ax2.tick_params('y', colors='r')

    plt.legend()
    fig.tight_layout()
    plt.show();