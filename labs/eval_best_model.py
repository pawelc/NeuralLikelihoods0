import os


# python -m run_exp -data_set inv_sin -model rnade_laplace  2>&1  | tee run_exp.log
from asynch import invoke_in_process_pool, Callable
from experiment.factory import  create_load_experiment
from flags import FLAGS

def run():
    target_dir = os.path.join(FLAGS.dir, FLAGS.data_set)
    os.makedirs(target_dir, exist_ok=True)
    os.chdir(target_dir)
    exp = create_load_experiment()
    exp.eval_best_model_and_save()

if __name__ == '__main__':
    invoke_in_process_pool(1, Callable(run))
