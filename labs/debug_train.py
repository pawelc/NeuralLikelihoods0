import json
import os

import experiment.factory


#  python -m debug_train -data_set sin -model mdn -args '' -print_tensors -print_tensors_filter W -print_tensors_summarize 10  2>&1  | tee debug.log
from data.registry import create_data_loader
from flags import FLAGS

if __name__ == '__main__':
    target_folder = os.path.join(FLAGS.dir, FLAGS.data_set)
    os.makedirs(target_folder, exist_ok=True)
    os.chdir(target_folder)
    data_loader = create_data_loader()
    train_eval_model_factory = getattr(experiment.factory, FLAGS.model + "_train_eval_model_factory")(data_loader)

    train_eval_model = train_eval_model_factory.create_train_eval(999)

    args = json.loads(FLAGS.args)

    train_eval_model(**args)