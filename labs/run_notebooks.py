import os

from experiment.experiment_vis import create_vis_notebooks, create_compare_notebooks,create_compare_all_experiments_notebook

if __name__ == '__main__':
    print("create_compare_all_experiments_notebook")
    create_compare_all_experiments_notebook()
    print("create_compare_notebooks")
    create_compare_notebooks()
    print("create_vis_notebooks")
    create_vis_notebooks()
