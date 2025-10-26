import torch ### DO NOT REMOVE THIS !!!!!

import os
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from tensorflow.keras.models import Sequential, Model, load_model
import qkeras
from qkeras import QBatchNormalization
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
import tensorflow as tf
import h5py as h5
import numpy as np
from tensorflow import keras 
from tqdm import tqdm

from losses import *
from model import *
from utilities import *
from optim import *

import wandb
import ray
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray import tune
import random

import gc
import argparse

def distance(model, data):
    mean, _ , _= model.encoder(data)
    score = np.sum(mean**2, axis=1)    
    return score

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
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    wandb.login(key="redacted")

    wandb.init(project = "Redacted",
               settings=wandb.Settings(_disable_stats=True),
               config = config)
    
    run_name = wandb.run.name
    
    beta = config['beta']
    VAE_lr = config['vae_lr']
    alpha = config['alpha']
    
    reco_scale = alpha * (1 - beta)
    kl_scale = beta
    
    device = 'cuda:0'  # This will be set later
    
    
    Epochs_VAE = 480
    Batch_size = 16384
    
    
    vae_encoder_nodes = config['vae_nodes']
    vae_latent_dim = config['vae_latent']
    
    # --------------------------------------------------------------------------
    
    f = h5.File('<redacted>', 'r')
    
    x_train = f['Background_data']['Train']['DATA'][:]
    x_test = f['Background_data']['Test']['DATA'][:]
    
    x_train_background = np.reshape(x_train, (x_train.shape[0], -1))
    x_test_background = np.reshape(x_test, (x_test.shape[0], -1))
    
    scale = f['Normalisation']['norm_scale'][:]
    bias = f['Normalisation']['norm_bias'][:]
    
    l1_bits_bkg_test = f['Background_data']['Test']['L1bits'][:]
    
    # --------------------------------------------------------------------------
    dataset = x_train_background
    dataset_test = x_test_background
    
    gc.collect()
    
    # --------------------------------------------------------------------------
    mask = {
    "MET":[True],
    "EGAMMA":[True,True,True,True,False,False,
              False,False,False,False,False,False],     
    "MUON":[True,True,True,True,False,False,False,False],
    "JET":[True,True,True,True,True,True,True,True,True,True,False,False]}
    
    
    loss_reco = cyl_PtPz_mae_loss(norm_scales = scale,
                         norm_biases = bias,
                         mask = mask)
    
    config_to_vae = {
        "encoder_config": {"nodes":vae_encoder_nodes},
        "latent_dim":vae_latent_dim,
        "decoder_config":{"nodes":[24,32,64,128,57]},
        "features":57,
        "ap_fixed_kernel":ap_fixed_kernel,
        "ap_fixed_bias":ap_fixed_bias,
        "ap_fixed_activation":ap_fixed_act,
        "ap_fixed_data":ap_fixed_data,
        "alpha":alpha,
        "beta":beta,
            
        
    }
    
    
    model = VariationalAutoEncoder_legacy(
    config=config_to_vae,
    reco_loss=loss_reco,
    kld_loss=kld())
    
    optimizer = lion(learning_rate=VAE_lr)
    model.compile(optimizer=optimizer)
    scheduler = cosine_annealing_warm_restart_with_warmup(first_cycle_steps = 32,
                                                              cycle_mult = 2,
                                                              max_lr=VAE_lr,
                                                              warmup_epochs=10,
                                                              gamma=0.5)
    
        
    #############################################
    # Signal Data
    #############################################
    SIGNAL_NAMES = list(f['Signal_data'].keys())
    
    signal_data_dict = {}
    signal_l1_dict = {}
    
    for signal_name in SIGNAL_NAMES:
        x_signal = f['Signal_data'][signal_name]['DATA'][:]
        x_signal = np.reshape(x_signal, (x_signal.shape[0], -1))
        l1_bits = f['Signal_data'][signal_name]['L1bits'][:]
    
        signal_data_dict[signal_name] = x_signal
        signal_l1_dict[signal_name] = l1_bits
    f.close()
        
        
    for present_epoch in tqdm(range(0, Epochs_VAE, 1)):
    
        train_loss = 0
        train_steps = 0
    
        index = torch.randperm(dataset.shape[0]).numpy()
        
        current_lr = scheduler.step()
        model.optimizer.learning_rate.assign(current_lr)
        for i in range(dataset.shape[0] // Batch_size):
            batch = dataset[index[i * Batch_size : (i + 1) * Batch_size]]
    
            model.train_step([batch,batch])
        
        
            
        metric = fast_score(
            model=model,
            data_bkg=dataset_test,
            bkg_l1_bits=l1_bits_bkg_test,
            distance_func=distance,
            data_signal=signal_data_dict,
            signal_l1_bits=signal_l1_dict,
            evaluation_threshold=1,
        )
    
        total_loss = model.total_loss_tracker.result().numpy()
        reco_loss = model.reconstruction_loss_tracker.result().numpy()
        kl_loss = model.kl_loss_tracker.result().numpy()
    
        
        # Reset metrics for next epoch
        model.total_loss_tracker.reset_states()
        model.reconstruction_loss_tracker.reset_states()
        model.kl_loss_tracker.reset_states()
    
        metric['EpochVae'] = present_epoch
        metric['LrVae'] = current_lr
        metric['TotalLossVae'] = total_loss
        metric['RecoLossVae'] = reco_loss
        metric['KLLossVae'] = kl_loss
        
        wandb.log(metric)
        ray.train.report(metrics=metric)

if __name__ == '__main__':
    
    ap_fixed_kernel = [6,2] ### To be further tuned !!!!
    ap_fixed_bias = [10,6] ### To be further tuned !!!!
    ap_fixed_act = [10,6] ### To be further tuned !!!!
    ap_fixed_data = [8,5] ### To be further tuned !!!!

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', type=str, default=None)

    args = parser.parse_args()

    if args.address:
        ray.init(address=args.address)
    else:
        ray.init(address='auto')

    search_space = {
        'vae_lr': tune.loguniform(5e-5, 5e-4),
        'beta': tune.uniform(0, 1),
        'alpha': tune.uniform(0, 1),
        'vae_latent': tune.sample_from(lambda spec: tune.randint(4, 12).sample()),
        'vae_nodes': tune.sample_from(lambda spec: [tune.randint(24, 32).sample(), tune.randint(10, 18).sample()]),
    }

    optuna_search = OptunaSearch(
        metric='raw-pure/haa4b-ma15',
        mode='max',
    )

    scheduler = ASHAScheduler(
        metric='raw-pure/haa4b-ma15',
        mode='max',
        max_t=480,
        grace_period=32,
        reduction_factor=2,
    )

    analysis = tune.run(
        run,
        config=search_space,
        storage_path='<redacted>',
        search_alg=optuna_search,
        scheduler=scheduler,
        num_samples=1000,
        resources_per_trial={'cpu': 8, 'gpu': 1 / 4}
    
    )
