# GL

## Setup

Clone this repo.  

(Optional) Create a virtual env with Python 3.11 and enter it:  

`python3.11 -m venv ~/.env/env`  

`source ~/.env/env/bin/activate`  

Install Python requirements  

`pip install -r requirements.txt`  

## Usage

1. Launch an Experiment.  
2. Review results. Cross-validation holdout results will be available in the latest numbered folder inside ```[cover,heart]/log```.   
3. Tensorboard summaries for training and validation learning curves will also be available and viewable with ```tensorboard --logdir=[cover,heart]/log```.  

## Experiments

Please refer to the paper for additional details, in particular the Appendix section.
There we describe the experimental settings, dataset construction, model used, training details, and provide numerical results.

**Main (entrypoint)**: `run.py`  
**FedAvg Implementation**: `federated.py`  
**Training Function**: `train.py`  

### Covertype

**Experiment Directory (in this repo)**: `cover`  
**Experiment Results**: `cover/log`  
**Experiment Results From Paper**: `cover/log/in-paper` and `cover/log/in-paper-centralized`   
**Dataloader**: `cover/dataset.py`  
**Model**: `cover/models/model.py`  

#### Original Dataset Source

The datasets have already been downloaded, but you may re-download them if you wish and place in `cover/data/`.  

**Original Dataset Source**: https://archive.ics.uci.edu/ml/datasets/covertype  

#### Prerequisite Data Preparation

First convert to the 11 feature version  

`python datasets/cover/convert_cover54_to_cover11.py --path-to-dataset datasets/cover/covtype.data --output-folder datasets/cover/ --verbose True`  

Next partition into 4 regional clients with train/test/validation splits  

`python datasets/cover/partition.py`  

#### Launch Command

`python -W ignore::UserWarning run.py --device cpu --epochs 105 --model model --batches 10 --learning-rate 0.001 --weight-decay 0.0001 --experiment cover --seed-range 2:28`

For Centralized baseline, also run:  

`python -W ignore::UserWarning run.py --device cpu --epochs 105 --model model --batches 10 --learning-rate 0.001 --weight-decay 0.0001 --experiment cover --seed-range 2:28 --centralized`  

Note, the `ignore::UserWarning` supresses this specific torch warning `UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/NestedTensorImpl.cpp:180` which is otherwise continuously triggered as we use Torch NestedTensor for our FedAvg implementation. You can omit this if you wish. 

### Heart Disease

**Experiment Directory (in this repo)**: `heart`  
**Experiment Results**: `heart/log`  
**Experiment Results From Paper**: `heart/log/in-paper`  
**Dataloader**: `heart/dataset.py`  
**Model**: `heart/models/model.py`  

#### Original Dataset Sources

The datasets have already been downloaded, but you may re-download them if you wish and place in `heart/data/`.  

**UCI Heart Disease (Cleveland)**: https://archive.ics.uci.edu/ml/datasets/Heart+Disease  
**South Africa Heart Disease**: http://statweb.stanford.edu/~tibs/ElemStatLearn/data.html  
**UCI Heart Disease (Cleveland)**: https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records  

#### Prerequisite Data Preparation

Although there is the following file which can be run to create train/test/validation splits  

`python3 heart/prepare_heart_disease.py`  

This logic is done as part of dataset loading based on the random seed for that particular run that you provided in the launch command. These will appear as subfolders with numeric names corresponding to the random seed for that run.  

#### Launch Command

`python -W ignore::UserWarning run.py --device cpu --epochs 10 --model model --batches 5 --learning-rate 0.001 --weight-decay 0.0001 --experiment heart --seed-range 0:100`

## Results

A number of CSV files will be created

`[cover,heart]/log/[run_number]/summary` will contain summaries of each metric averaged across all seeded cross-validation runs.

`[cover,heart]/log/[run_number]/` will contain 2 files for each comparison method:

a) one with *summary* at the end; this has additional statistics besides average over seeded cross-validation runs  
b) the other file contains metrics in time-series format across all epochs  

and  

there will also be individual subfolders with names `seed_[random_seed]` for each seeded run, containing Tensorboard summaries as well as detailed run statistics including confusion matrix.  