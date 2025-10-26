###Change log

2023-05-31 : update to latest RUN-3 training, updated to latest hls4ml.

# DNN and CNN autoenconder

Training and testing code for DNN and CNN autoencoders with CMS L1 emulator inputs.

### Setup environment

Install [miniconda3](https://repo.anaconda.com/miniconda/) (check website for latest version):

```
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
bash ./Miniconda3-py39_4.10.3-Linux-x86_64.sh
```

Clone this repository:

```
git clone https://gitlab.cern.ch/l1a/l1_anomaly_ae.git
cd l1_anomaly_ae
```

Setup the conda environment

```
conda env create --file=environment.yml
```


Activate the environment (nb, every time you login):

```
conda activate l1ad
```

### Train and evaluate DNN models:

The following script takes as input the h5 files with GT L1 info ([files name and location](https://docs.google.com/document/d/12UVyXYwy92qwYeEsdN_nreV_EohzgavD6s7kSrSKn-w/edit?usp=sharing)) and run the training + performance evaluation workflow.
Files location and model or training options are specified in a configuration yaml file (as config.yml). Other options are passed as arguments. To run the full workflow:

```
python dnn/end2end.py --run all --config dnn/config-PU65.yml
```

The script saves the trained model and the results/plots from the evaluation step in the output directory specified in the yaml configuration file.

Script options:

1) To run the workflow with QKeras, make sure to add the flag ```--model_quantize``` for the above script

2) The option `--run` can be set to `train` or `eval` if one wants to run only one of the two steps.

3) The training step first dumps the data into a pickle file with name specified in the yaml configuration file. The file is saved in the same directory from which the script is run. If one already has this file, this step can be skipped with the `---load_pickle` option.

4) The evaluation step first dumps the predictions from the training into the results.h5 file in the output directory specified in the in the yaml configuration file. If nothing changes in the trained model, this step can be skipped at following iterations with the `---load_results` option.

This script is also available for an interactive run in [```End2End_demo.ipynb```](https://gitlab.cern.ch/cms-l1-ad/l1_anomaly_ae/-/blob/master/dnn/End2End_demo.ipynb). 

### Firmware

High Level Synthesis code generated from the Neural Network using hls4ml goes to the repository: https://gitlab.cern.ch/ssummers/run3_ugt_ml 
The firmware is compiled into a Python module called `ugt_hls_emulator` that is used for validation.

### Train and evaluate CNN models: (Depricated)

#### Prepare the data

Input data must be prepared with ```prepare_data.py```, which accepts as inputs preprocessed QCD and/or preprocessed BSM h5 files and outputs a pickle file with train/test events split to be used downstream for the train and evaluation steps. For example: 

```
python prepare_data.py --input-file QCD_preprocessed.h5 --input-bsm BSM_preprocessed.h5 --output-file data.pickle
```

nb, this script has the list of BSM signals [hardcoded](https://gitlab.cern.ch/cms-l1-ad/l1_anomaly_ae/-/blob/master/cnn/prepare_data.py#L37-L52) and you might want to change if new signals will be added or if unused signals will be removed.

#### Training 

The training program requires the input of the latent dimension, filenames for the output model (h5 and json), the training data pickle file, the history, but also the batch size, and the number of epochs. For example: 

```
python train.py --latent-dim 4 --output-model-h5 output_model.h5 --output-model-json output_model.json --input-data data.pickle --output-history history.h5 --batch-size 256 --n-epochs 30
```

Two models can be trained: Conv VAE or Conv AE. The model can be set with `--model-type` as either `conv_vae` or `conv_ae`. Change all other options as needed.

#### Performance evaluation

First prepare file with predictions for the QCD test sample and BSM samples:

```
python evaluate.py --input-h5 output_model.h5 --input-json output_model.json --input-history history.h5 --output-result result.h5 --input-file data.pickle
```

Make ROC curves, history and loss distributions:

```
python plot.py --coord cyl --model vae --loss-type mse_kl --output-dir ./ --input-dir ./ --label my_vae
```

Several loss types can be set and type of features. Change all other options as needed.
