find . -type f -name "*.pkl" | xargs -i cp --parents "{}" ../../RM/labs/results/
find . -type f -name "*.ipynb" | xargs -i cp --parents "{}" ../../RM/labs/results/