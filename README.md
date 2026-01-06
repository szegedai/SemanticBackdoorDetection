## Environment setup
* `conda create -n env_sbd python=3.9`
* `conda activate env_sbd`
* `conda install pytorch==2.0.0 torchvision==0.15 pytorch-cuda=11.7 -c pytorch -c nvidia`
* `pip install git+https://github.com/RobustBench/robustbench.git`
* `pip install "numpy<2" matplotlib einops scipy scikit-image jupyter`

## Model training
See `multi_model_train.sh`.
For training models with multi-class poisoning attacks, see the readme in `multi-class`.

## Inverted distance test set generation
See the readme in `create_inverted_test_set`.

## Measurements

`run_model_x_eval.sh` measures distance between pairs of models in a pool, and creates a result file that can be analysed afterwards.
At least 2 clean and 2 poisoned models are needed. (Correct label is determined from name by function `backdoor` in `run_model_x_eval.sh` and `voting6.py`.)

To analyse result files:
* `cross_val.sh` performs cross-validated experiments
* you need to modify parameters in `voting6.py` to use separate train and test sets
* `matrix.sh` was used for experiments in section Generalization Over Attack Types
