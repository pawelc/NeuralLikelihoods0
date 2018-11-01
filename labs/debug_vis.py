import os

from experiment.experiment_vis import ExperimentVis


#  python -m debug_vis -data_set inv_sin -model nn_pdf -print_tensors -print_tensors_filter loss -print_tensors_summarize 10 -debug_tb  2>&1  | tee debug.log
from flags import FLAGS

if __name__ == '__main__':
    os.chdir(os.path.join(FLAGS.dir,FLAGS.data_set))
    exp = ExperimentVis()
    samples = exp.eval_best_model(["train"])