import torch
import sys
import numpy as np
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
import h5py

import tensorflow as tf


K = tf.keras.backend

from axo import data_util
from axo import model
from axo import metric
from axo import utilities
from axo import recipies

import axo

import gc
import re
import pprint
import json

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Input
from tensorflow.keras.models import Sequential, Model
import mplhep as hep
from tqdm.auto import tqdm
import argparse
import yaml
import os

#######################################################################################################
# Fixing random seed
#######################################################################################################
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)    

os.environ["PYTHONHASHSEED"] = str(seed)
tf.random.set_seed(seed)
tf.config.experimental.enable_op_determinism()
#######################################################################################################

def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))
yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)

def main(slave = {}, ray_backend = False):
    #######################################################################################################
    # Setting TF Memory Growth
    #######################################################################################################
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    ########################################################################################################
    # Dictionary merging and config checking
    ########################################################################################################
    module_dir = os.path.dirname(__file__)

    master = yaml.load(open(os.path.join(module_dir,"utilities/config.yml"),"r"), Loader=yaml.Loader)

    if utilities.check_compartibility(master=master,slave=slave) == 1:
        print("Configurations are compatible")
        print("Generating new config ....")
        daughter = utilities.merge_dict(master=master,slave=slave)
    else:
        print("Smoke test failed !!! check the dictionary and the documentations")
        return 0
    config = daughter.copy()  # Setting Daughter as default

    DEBUG = config["DEBUG"] # Adding Debug flag
    
    if DEBUG:
        print("WARNING !!!!, DEBUG Flag found....")

    ########################################################################################################
    # Data creation
    ########################################################################################################
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

    ########################################################################################################
    # Reproducibility
    ########################################################################################################
    seed = config["determinism"]["global_seed"]
    if config["determinism"]["python_determinism"]:
        os.environ["PYTHONHASHSEED"] = str(seed)
    if config["determinism"]["numpy_determinism"]:
        np.random.seed(seed)
    if config["determinism"]["tf_op_determinism"]:
        tf.random.set_seed(seed)
        tf.config.experimental.enable_op_determinism()
    if config["determinism"]["torch_manual_seed"]:
        torch.manual_seed(seed)
    if config["determinism"]["torch_cuda_manual_seed"]:
        torch.cuda.manual_seed(seed)
    
    ########################################################################################################
    # Data creation
    ########################################################################################################

    if not ray_backend:
        print("Ray backend not found preparing the dataset")
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
   
    ########################################################################################################
    # To model recipe
    ########################################################################################################
    compute_recipe = getattr(axo.recipies, config["train_recipe"])
    if ray_backend:
        compute_recipe(config, DEBUG=DEBUG, ray_backend=ray_backend)
        print("Ray backend found,skipping further steps")
        return
    else:
        print("Ray backend not found, proceeding with all steps")
        complete_model, enc, vae, history = compute_recipe(config, DEBUG=DEBUG) 
    ########################################################################################################
    # Score plots and result generation
    ########################################################################################################
    config_thres = config["threshold"]
    config_thres["data_path"] = config["data_config"]["Processed_data_path"]
    axo_man = axo.metric.axo_threshold_manager(model=complete_model, config=config_thres)
    axo_embed = axo.metric.axo_embedding(model=complete_model,
                          enc=enc,
                          vae=vae,
                          config=config,
                          axo_manager=axo_man)
    dist_plot = axo.metric.distribution_plots(model=complete_model, config=config_thres)

    axo_man.get_threshold_single_nu()
    ########################################################################################################
    # Storage and report generation
    ########################################################################################################
    if DEBUG:
        print("DEBUG Flag found!!!, only storing lite.h5 and build.h5")
        
    utilities.store_axo(
        config=config,
        model=complete_model,
        axo_man=axo_man,
        dist_plot=dist_plot,
        history_dict=history,
        DEBUG = DEBUG
    )
    
    threshold_dict  = utilities.retrieve.get_threshold_dict(config["store"]["lite_path"])
    dict_axo = utilities.retrieve.get_axo_score_dataframes(config["store"]["lite_path"])
    histogram_dict = utilities.retrieve.get_histogram_dict(config["store"]["lite_path"])
    history_dict = utilities.retrieve.get_history_dict(config["store"]["lite_path"])
    raw_wrt_pure = utilities.retrieve.get_raw_wrt_pure(config["store"]["lite_path"])
    
    report_config = config["report"]

    generate_report_flag = report_config["generate"]
    if DEBUG:
        print("DEBUG Flag found skipping report generation")
        generate_report_flag = False
    
    if generate_report_flag:
        print("Report generation flag found !!")
        html_path = report_config["path"]
        utilities.generate_axolotl_html_report(
            config=config,
            all_axo = axo_man,
            axo_embed = axo_embed,
            dist_plots = dist_plot,
            dict_axo=dict_axo,
            histogram_dict=histogram_dict,
            threshold_dict=threshold_dict,
            history_dict=history_dict,
            raw_wrt_pure=raw_wrt_pure,
            output_file=html_path,
            DEBUG = DEBUG
        )
 
    
    print("Run completing exiting ...")
    
