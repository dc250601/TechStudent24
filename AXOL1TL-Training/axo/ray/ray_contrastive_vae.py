import ray
import sys
import os
from os.path import dirname, abspath, join
sys.path.append(abspath(join(dirname(__file__), "..","..")))
import axo
from axo.model import *
from axo.utilities import *
from axo import *
from axo import data_util
import random
from collections import defaultdict
import argparse
import ray
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray import tune
import yaml
import os
import torch
import numpy as np
import tensorflow as tf

def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))
yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)

def run(config):
    ###################################################################
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)    

    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()

    ####################################################################

    axo_root_location = os.path.join(os.path.dirname(__file__), "..")
    cfg_path = os.path.join(axo_root_location, "config_library", "v5_ray_template.yml")
    cfg = yaml.load(open(cfg_path,"r"), Loader=yaml.Loader)
    cfg["train"]["Contrastive_VAE"]["optimiser_config_contrastive"]["learning_rate"] = config["vic_lr"]
    cfg["train"]["Contrastive_VAE"]["optimiser_config_vae"]["learning_rate"] = config["vae_lr"]
    cfg["train"]["Contrastive_VAE"]["blur_p"] = config["blur_p"]
    cfg["train"]["Contrastive_VAE"]["blur_m"] = config["blur_m"]
    cfg["train"]["Contrastive_VAE"]["blur_s"] = config["blur_s"]
    cfg["train"]["Contrastive_VAE"]["mask_p"] = config["mask_p"]
    cfg["model"]["train_mode"]["Contrastive_VAE"]["beta"] = config["beta"]
    cfg["model"]["train_mode"]["Contrastive_VAE"]["alpha"] = config["alpha"]
    cfg["model"]["train_mode"]["Contrastive_VAE"]["encoder_nodes"] = config["encoder_nodes"]
    cfg["model"]["train_mode"]["Contrastive_VAE"]["vae_latent"] = [config["vae_latent"]]
    cfg["model"]["train_mode"]["Contrastive_VAE"]["vae_nodes"] = config["vae_nodes"]
    axo.recipies.contrastive_vae.run(config=cfg,ray_backend=True)


def dataset_creator(config):

    processed_data_path = config["data_config"]["Processed_data_path"]
    if os.path.isfile(processed_data_path):
        print("File already exists, checking if the config match")
        f = h5py.File(config["data_config"]["Processed_data_path"], "r")
        exisitng_ser_config = f.attrs["config"]
        f.close()
        present_ser_config = json.dumps(config["data_config"])

        # Uncomment the following lines if you want to regenerate data when configs don't match
        if exisitng_ser_config == present_ser_config:
             print("Configs match, skipping")
        else:
             print("[WARNING]: CONFIG DO NOT MATCH, OVERWRITING!!!")
             data_util.data.get_data(config_master=config["data_config"])
    else:
         print("File does not exist, creating data file")
         data_util.data.get_data(config_master=config["data_config"])


if __name__ == "__main__":

    yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)
    parser = argparse.ArgumentParser(description="Contrastive VAE Training")

    axo_root_location = os.path.join(os.path.dirname(__file__), "..")

    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    parser.add_argument("--ZeroBiasPath", type=str, required=True, help="Path to the zerobias dataset")
    parser.add_argument("--BSMPath", type=str, required=True, help="Path to the BSM dataset")
    parser.add_argument("--ExperimentName", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--dataConfig", type=str, default=os.path.join(axo_root_location,"config_library","vanilla_data96.yml"), help="Path to the data config file")
    parser.add_argument("--ray_runs", type=int, default=2000, help="Number of Ray iterations")
    parser.add_argument("--experiment_path", type=str, default="./", help="Path to the Experiment directory")
    parser.add_argument("--NumberofCPUs", type=int, default=4, help="Number of CPUs per RAY worker")
    parser.add_argument("--NumberofGPUs", type=int, default=1/8, help="Number of GPUs per RAY worker")
    parser.add_argument("--ray_address", type=str, default=None, help="Address of the Ray cluster (if running on a cluster)")
    args = parser.parse_args()

    # Load the config file
    data_config = yaml.load(open(args.dataConfig,"r"), Loader=yaml.Loader)
    data_config["data_config"]["Read_configs"]["BACKGROUND"]["file_path"] = args.ZeroBiasPath
    data_config["data_config"]["Read_configs"]["SIGNAL"]["file_path"] = args.BSMPath
    
    os.makedirs(os.path.join(args.experiment_path, args.ExperimentName), exist_ok=True)
    data_config["data_config"]["Processed_data_path"] = os.path.join(args.experiment_path, args.ExperimentName, "Data.h5")
    dataset_creator(data_config)




    # if args.ray_address:
    #     ray.init(address=args.ray_address)
    # else:
    #     ray.init(address='auto')

    # search_space = {
    #     'vic_lr': tune.loguniform(1e-4, 1e-3),
    #     'vae_lr': tune.loguniform(1e-4, 1e-3),
    #     'blur_p': tune.uniform(0, 1),
    #     'blur_m': tune.uniform(0, 1),
    #     'blur_s': tune.uniform(0, 1),
    #     'mask_p': tune.uniform(0, 1),
    #     'beta': tune.uniform(0, 1),
    #     'alpha': tune.uniform(0, 1),
    #     'encoder_nodes': tune.sample_from(lambda spec: [tune.randint(24, 32).sample(), tune.randint(8, 18).sample()]),
    #     'vae_latent': tune.sample_from(lambda spec: tune.randint(4, 6).sample()),
    #     'vae_nodes': tune.sample_from(lambda spec: [tune.randint(8, 12).sample(), tune.randint(6, 8).sample()]),
    # }

    debug_search_space = {
        'vic_lr': 0.0001,
        'vae_lr': 0.0001,
        'blur_p': 0.5,
        'blur_m': 0.5,
        'blur_s': 0.5,
        'mask_p': 0.5,
        'beta': 0.5,
        'alpha': 0.5,
        'encoder_nodes': [28, 12],
        'vae_latent': 5,
        'vae_nodes': [10, 6],
    }

    # optuna_search = OptunaSearch(
    #     metric='raw-pure/haa4b-ma15',
    #     mode='max',
    # )

    # scheduler = ASHAScheduler(
    #     metric='raw-pure/haa4b-ma15',
    #     mode='max',
    #     max_t=480,
    #     grace_period=32,
    #     reduction_factor=2,
    # )

    # analysis = tune.run(
    #     run,
    #     config=search_space,
    #     storage_path= os.path.join(args.experiment_path, args.ExperimentName),
    #     search_alg=optuna_search,
    #     scheduler=scheduler,
    #     num_samples=args.ray_runs,
    #     resources_per_trial={'cpu': args.NumberofCPUs, 'gpu': args.NumberofGPUs},

    # )

    run(debug_search_space)






