Notebooks to perform inference on anomalous diffusion trajectories in the format of the ones produced by the AnDi Challenge datasets.
The trajectories have to be generated or downloaded.
The folds nets contains the neural networks to be used.
The folder predictions contains the predictions made on the challenge data using the notebooks and the nets.
The folder best_sub contains the predictions submitted in the challenge phase that performed the best.

The notebook task1_submission.ipynb performs the predictions for task 1.
The notebook task1_collating results.ipynb combines the predictions and prints a file task1.txt in the format required for the challenge submission.

The notebook task2_submission.ipynb performs the predictions for task 2.
The notebook task2_collating results.ipynb combines the predictions and prints a file task2.txt in the format required for the challenge submission.

The notebook task3_submission.ipynb performs the predictions for task 3 and prints a file task3.txt in the format required for the challenge submission.

The files data_split.py and many_net.py contain the functions required for the evaluation of the notebooks.


