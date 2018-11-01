import numpy as np

import data
import data.registry
from data.tf_gen_data import MPGFactory, SinusoidFactory
from flags import FLAGS


def sin():
    data_loader = data.TfGenerator(data.Config(dir='data').add_param('samples', 10000).
                                   add_param("op_factory", SinusoidFactory("normal")).
                                   add_param("x", np.reshape(np.random.uniform(-1.5, 1.5, 10000), (-1, 1))))
    data_loader.load_data()
    return data_loader


def sin_t_noise():
    data_loader = data.TfGenerator(data.Config(dir='data').add_param('samples', 10000).
                                   add_param("op_factory", SinusoidFactory("standard_t")).
                                   add_param("x", np.reshape(np.random.uniform(-1.5, 1.5, 10000), (-1, 1))))
    data_loader.load_data()
    return data_loader


def inv_sin():
    data_loader = data.TrendingSinusoid(data.Config('data', normalize=True))
    data_loader.load_data()
    return data_loader


def inv_sin_t_noise():
    data_loader = data.TrendingSinusoid(data.Config('data', normalize=True).add_param("noise", "standard_t").
                                        add_param("df", 3))
    data_loader.load_data()
    return data_loader


def etf():
    data_loader = data.Yahoo(
        data.Config('data', normalize=True).add_param('symbols', ["SPY"]).
            add_param('start', "2011-01-03").add_param('end', "2015-04-14"))
    data_loader.load_data()
    return data_loader


def etf2d():
    data_loader = data.Yahoo(
        data.Config('data', normalize=True, x_slice=slice(None, -2), y_slice=slice(-2, None)).
            add_param('symbols', ["SPY", "DIA"]).
            add_param('start', "2011-01-03").
            add_param('end', "2015-04-14"))
    data_loader.load_data()
    return data_loader


def uci_redwine():
    data_loader = data.UCI(data.Config(dir='data', normalize=True, x_slice=slice(None, -2), y_slice=slice(-2, None), uniquenessThreshold=0.05).
                           add_param('file', "winequality-red.csv").add_param('delimiter', ';'))
    data_loader.load_data()
    return data_loader


def uci_whitewine():
    data_loader = data.UCI(data.Config(dir='data', normalize=True, x_slice=slice(None, -2), y_slice=slice(-2, None), uniquenessThreshold=0.05).
                           add_param('file', "winequality-white.csv").add_param('delimiter', ';'))
    data_loader.load_data()
    return data_loader


def uci_parkinsons():
    data_loader = data.UCI(data.Config(dir='data', normalize=True, x_slice=slice(None, -2), y_slice=slice(-2, None), uniquenessThreshold=0.05).
                           add_param('file', "parkinsons_updrs_processed.data").add_param('delimiter', ','))
    data_loader.load_data()
    return data_loader


def uci_redwine_joint():
    data_loader = data.UCI(data.Config(dir='data', normalize=True, x_slice=slice(0), y_slice=slice(None), uniquenessThreshold=0.05).
                           add_param('file', "winequality-red.csv").add_param('delimiter', ';'))
    data_loader.load_data()
    return data_loader


def uci_whitewine_joint():
    data_loader = data.UCI(data.Config(dir='data', normalize=True, x_slice=slice(0), y_slice=slice(None), uniquenessThreshold=0.05).
                           add_param('file', "winequality-white.csv").add_param('delimiter', ';'))
    data_loader.load_data()
    return data_loader


def uci_parkinsons_joint():
    data_loader = data.UCI(data.Config(dir='data', normalize=True, x_slice=slice(0), y_slice=slice(None), uniquenessThreshold=0.05).
                           add_param('file', "parkinsons_updrs_processed.data").add_param('delimiter', ','))
    data_loader.load_data()
    return data_loader


def mpg():
    data_loader = data.TfGenerator(data.Config(dir='data', x_slice=slice(None, -2), y_slice=slice(-2, None)).
                                   add_param('samples', 10000).
                                   add_param("op_factory", MPGFactory()).
                                   add_param("x", np.reshape(np.random.uniform(-10, 10, 10000), (-1, 1))))
    data_loader.load_data()
    return data_loader

FX_SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'NZDUSD','NZDJPY','GBPJPY']


def fx_all_predicted():
    data_loader = data.Fxcm(data.Config('data',normalize=True,x_slice=slice(None, -8), y_slice=slice(-8, None)).add_param('symbols',
                                                                                                           FX_SYMBOLS)
                            .add_param('start', '2015-01-05')
                            .add_param('end', '2015-01-30')
                            .add_param('ar_terms', 2))
    data_loader.load_data()
    return data_loader

def fx_eurgbp_predicted():
    data_loader = data.Fxcm(data.Config('data',normalize=True,x_slice=slice(None, -2), y_slice=slice(-2, None)).add_param('symbols',FX_SYMBOLS)
                            .add_param('start', '2015-01-05')
                            .add_param('end', '2015-01-30')
                            .add_param('ar_terms', 4).add_param('predicted_idx', [FX_SYMBOLS.index('EURUSD'),FX_SYMBOLS.index('GBPUSD')]))
    data_loader.load_data()
    return data_loader

def fx_eur_predicted():
    data_loader = data.Fxcm(data.Config('data',normalize=True,x_slice=slice(None, -1), y_slice=slice(-1, None)).add_param('symbols',FX_SYMBOLS)
                            .add_param('start', '2015-01-05')
                            .add_param('end', '2015-01-30')
                            .add_param('ar_terms', 10).add_param('predicted_idx', [FX_SYMBOLS.index('EURUSD')]))
    data_loader.load_data()
    return data_loader

def create_data_loader():
    data_set_factory = getattr(data.registry, FLAGS.data_set)
    return data_set_factory()
