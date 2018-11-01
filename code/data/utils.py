import pickle
import os

from flags import FLAGS


def load_data_seeds():
    try:
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data_seed.pkl") , 'rb') as f:
            return pickle.load(f)
    except:
        return None

def store_data_seeds(data_seeds):
    os.makedirs(FLAGS.dir, exist_ok=True)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data_seed.pkl"), 'wb') as f:
        pickle.dump(data_seeds, f)