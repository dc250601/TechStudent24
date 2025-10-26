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

from model import *
from utilities import *

import wandb
import ray
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray import tune
import random

import gc
import argparse

def distance(model, data):
    mean, _ = model.encoder(data)
    reco = model.decoder(mean)

    score = tf.keras.losses.mean_squared_error(data,reco)
    
    return score.numpy()

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

    wandb.init(project = "redacted",
               settings=wandb.Settings(_disable_stats=True),
               config = config)
    
    run_name = wandb.run.name
    
    blur_p = config['blur_p']
    blur_m = config['blur_m']
    blur_s = config['blur_s']

    mask_p = config['mask_p']

    beta = config['beta']
    VIC_lr = config['vic_lr']
    VAE_lr = config['vae_lr']
    alpha = config['alpha']

    reco_scale = alpha * (1 - beta)
    kl_scale = beta

    device = 'cuda:0'  # This will be set later

    Epochs_contrastive = 50  # This will be set later
    
    Epochs_VAE = 480

    Batch_size = 4096

    vic_encoder_nodes = config['encoder_nodes']
    projector_features = vic_encoder_nodes[-1] * 4

    vae_encoder_nodes = config['vae_nodes']
    vae_latent_dim = config['vae_latent']

    # Making it symmetric

    vae_decoder_nodes = [vic_encoder_nodes[-1]] + vae_encoder_nodes.copy()
    vae_decoder_nodes.reverse()
    
    # --------------------------------------------------------------------------
    input_q = quantized_bits(ap_fixed_data[0], ap_fixed_data[1], alpha=1)
    # --------------------------------------------------------------------------
    
    f = h5.File('/pscratch/sd/d/diptarko/TECH-L1AD/V5WithBugFix/Data_unclipped.h5', 'r')

    x_train = input_q(f['Background_data']['Train']['DATA'][:])
    x_test = input_q(f['Background_data']['Test']['DATA'][:])

    x_train_background = np.reshape(x_train, (x_train.shape[0], -1))
    x_test_background = np.reshape(x_test, (x_test.shape[0], -1))

    scale = f['Normalisation']['norm_scale'][:]
    bias = f['Normalisation']['norm_bias'][:]

    l1_bits_bkg_test = f['Background_data']['Test']['L1bits'][:]
    

    feature_blur = FastFeatureBlur(p=blur_p, strength=blur_s, magnitude=blur_m, device=device)
    feature_blur_prime = FastFeatureBlur(p=blur_p, strength=blur_s, magnitude=blur_m, device=device)

    object_mask = FastObjectMask(p=mask_p, device=device)
    object_mask_prime = FastObjectMask(p=mask_p, device=device)

    lorentz_rot = FastLorentzRotation(p=0.5, norm_scale=scale, norm_bias=bias, device=device)
    lorentz_rot_prime = FastLorentzRotation(p=0.5, norm_scale=scale, norm_bias=bias, device=device)

    # --------------------------------------------------------------------------
    dataset = torch.tensor(x_train_background, dtype=torch.float32, device=device)
    dataset_test = torch.tensor(x_test_background, dtype=torch.float32, device=device)
    del x_train_background
    gc.collect()

    # --------------------------------------------------------------------------

    Backbone = ModelBackbone(nodes=vic_encoder_nodes,
                         ap_fixed_kernel = ap_fixed_kernel,
                         ap_fixed_bias= ap_fixed_bias,
                         ap_fixed_activation = ap_fixed_act)

    Projection = ModelProjector(projector_features)
    model = VICReg(backbone=Backbone, projector=Projection, num_features=projector_features, batch_size=Batch_size)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=VIC_lr))

    scheduler = cosine_with_warmup(
    max_lr=VIC_lr,
    warmup_epochs=10,
    decay_epochs=Epochs_contrastive - 10)

    for present_epoch in tqdm(range(0, Epochs_contrastive, 1)):
        train_loss = 0
        train_steps = 0
    
        index = torch.randperm(dataset.shape[0])
        
        current_lr = scheduler.step()
        model.optimizer.learning_rate.assign(current_lr)
        for i in range(dataset.shape[0] // Batch_size):
            with torch.no_grad():
                batch = dataset[index[i * Batch_size : (i + 1) * Batch_size]]
            
                batch_x = batch.clone()
                batch_y = batch.clone()
        
                batch_x = feature_blur(batch_x)
                batch_x = object_mask(batch_x)
                batch_x = lorentz_rot(batch_x)
        
                batch_y = feature_blur_prime(batch_y)
                batch_y = object_mask_prime(batch_y)
                batch_y = lorentz_rot_prime(batch_y)
    
            batch_x,batch_y = batch_x.cpu().numpy(),batch_y.cpu().numpy()
            metrics = model.train_step((batch_x, batch_y))
    
        epoch_loss = model.loss_tracker.result().numpy()
        epoch_repr = model.loss_tracker_repr.result().numpy()
        epoch_std = model.loss_tracker_std.result().numpy()
        epoch_cov = model.loss_tracker_cov.result().numpy()
        
        # Reset metrics for next epoch
        model.loss_tracker.reset_states()
        model.loss_tracker_repr.reset_states()
        model.loss_tracker_std.reset_states()
        model.loss_tracker_cov.reset_states()
    
        metric_embed = {}
        metric_embed['TrainLossC'] = epoch_loss
        metric_embed['EpochC'] = present_epoch
        metric_embed['LrC'] = current_lr
        
        wandb.log(metric_embed)
    
    #########################################################################################################################
    #########################################################################################################################
    #########################################################################################################################

    
    vic_encoder = model.backbone
    
    encoder = VAE_Encoder(nodes=vae_encoder_nodes,
                          feature_size=vae_latent_dim,
                          ap_fixed_kernel = ap_fixed_kernel,
                          ap_fixed_bias= ap_fixed_bias,
                          ap_fixed_activation = ap_fixed_act)
    decoder = VAE_Decoder(nodes=vae_decoder_nodes,
                          ap_fixed_kernel = ap_fixed_kernel,
                          ap_fixed_bias= ap_fixed_bias,
                          ap_fixed_activation = ap_fixed_act)
    
    model = VariationalAutoEncoder(encoder=encoder, decoder=decoder)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=VAE_lr))
    scheduler = cosine_annealing_warm_restart_with_warmup(first_cycle_steps = 32,
                                                          cycle_mult = 2,
                                                          max_lr=VAE_lr,
                                                          warmup_epochs=10,
                                                          gamma=0.5)
    
    dataset_latent = vic_encoder(dataset.cpu().numpy()).numpy()
    dataset_latent_test = vic_encoder(dataset_test.cpu().numpy()).numpy()
    
    #############################################
    # Signal Data
    #############################################
    SIGNAL_NAMES = list(f['Signal_data'].keys())
    
    signal_data_dict = {}
    signal_l1_dict = {}
    
    for signal_name in SIGNAL_NAMES:
        x_signal = input_q(f['Signal_data'][signal_name]['DATA'][:])
        x_signal = np.reshape(x_signal, (x_signal.shape[0], -1))
        x_signal = vic_encoder(x_signal).numpy()
        l1_bits = f['Signal_data'][signal_name]['L1bits'][:]
    
        signal_data_dict[signal_name] = x_signal
        signal_l1_dict[signal_name] = l1_bits
    f.close()
    
    
    for present_epoch in tqdm(range(0, Epochs_VAE, 1)):
    
        train_loss = 0
        train_steps = 0
    
        index = torch.randperm(dataset_latent.shape[0]).numpy()
        
        current_lr = scheduler.step()
        model.optimizer.learning_rate.assign(current_lr)
        for i in range(dataset_latent.shape[0] // Batch_size):
            batch = dataset_latent[index[i * Batch_size : (i + 1) * Batch_size]]
    
            model.train_step(batch)
        
        
            
        metric = fast_score(
            model=model,
            data_bkg=dataset_latent_test,
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
        # scheduler.step()  #### One of the original bugs


if __name__ == '__main__':
    
    ap_fixed_kernel = [6,2] ### To be further tuned !!!!
    ap_fixed_bias = [10,6] ### To be further tuned !!!!
    ap_fixed_act = [10,6] ### To be further tuned !!!!
    ap_fixed_data = [9,6] ### To be further tuned !!!!

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', type=str, default=None)

    args = parser.parse_args()

    if args.address:
        ray.init(address=args.address)
    else:
        ray.init(address='auto')

    search_space = {
        'vic_lr': tune.loguniform(1e-4, 1e-3),
        'vae_lr': tune.loguniform(1e-4, 1e-3),
        'blur_p': tune.uniform(0, 1),
        'blur_m': tune.uniform(0, 1),
        'blur_s': tune.uniform(0, 1),
        'mask_p': tune.uniform(0, 1),
        'beta': tune.uniform(0, 1),
        'alpha': tune.uniform(0, 1),
        'encoder_nodes': tune.sample_from(lambda spec: [tune.randint(24, 32).sample(), tune.randint(8, 18).sample()]),
        'vae_latent': tune.sample_from(lambda spec: tune.randint(4, 6).sample()),
        'vae_nodes': tune.sample_from(lambda spec: [tune.randint(8, 12).sample(), tune.randint(6, 8).sample()]),
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
        storage_path="<redacted>",
        search_alg=optuna_search,
        scheduler=scheduler,
        num_samples=2000,
        resources_per_trial={'cpu': 8, 'gpu': 1 / 4},
    
    )
