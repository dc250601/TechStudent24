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

from axo.model import *
from axo.utilities import *
from axo import *
import ray
import axo
import random
from collections import defaultdict


import gc
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,        # or any other level you prefer
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def distance(model, data):
    mean, _ = model.encoder(data)
    reco = model.decoder(mean)

    score = tf.keras.losses.mean_squared_error(data,reco)
    
    return score.numpy()

def run(config,ray_backend = False, DEBUG=False):
    
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)    

    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()

    ap_fixed_kernel = config["model"]["ap_fixed_kernel"]
    ap_fixed_bias = config["model"]["ap_fixed_bias"]
    ap_fixed_act = config["model"]["ap_fixed_activation"]
    ap_fixed_data = config["model"]["ap_fixed_data"]
    Quantization_configs:
    quantize_bits:

    input_q = quantized_bits(ap_fixed_data[0], ap_fixed_data[1], alpha=1)
    
    blur_p = config["train"]["Contrastive_VAE"]['blur_p']
    blur_m = config["train"]["Contrastive_VAE"]['blur_m']
    blur_s = config["train"]["Contrastive_VAE"]['blur_s']

    mask_p = config["train"]["Contrastive_VAE"]['mask_p']

    beta = config["model"]["train_mode"]["Contrastive_VAE"]['beta']
    alpha = config["model"]["train_mode"]["Contrastive_VAE"]['alpha']

    reco_scale = alpha * (1 - beta)
    kl_scale = beta

    device = "cpu"

    Epochs_contrastive = config["train"]["Contrastive_VAE"]["epochs_contrastive"]
    
    Epochs_VAE = config["train"]["Contrastive_VAE"]["epochs_vae"]

    Batch_size = config["train"]["Contrastive_VAE"]["batch_size"]

    vic_encoder_nodes = config["model"]["train_mode"]["Contrastive_VAE"]["encoder_nodes"]
    projector_features = vic_encoder_nodes[-1] * 4

    vae_encoder_nodes = config["model"]["train_mode"]["Contrastive_VAE"]["vae_nodes"]
    vae_latent_dim = config["model"]["train_mode"]["Contrastive_VAE"]["vae_latent"][0]

    # Making it symmetric

    vae_decoder_nodes = [vic_encoder_nodes[-1]] + vae_encoder_nodes.copy()
    vae_decoder_nodes.reverse()

    # --------------------------------------------------------------------------
    history = {}

    _tracker_total_loss_vicreg = []
    _tracker_repr_loss_vicreg = []
    _tracker_std_loss_vicreg = []
    _tracker_std_cov_vicreg = []
    _tracker_epoch_contrastive = []
    _tracker_lr_contrastive = []

    _tracker_epoch_vae = []
    _tracker_lr_vae = []
    _tracker_reco_loss_vae = []
    _tracker_kld_loss_vae = []
    _tracker_total_vae_loss = []
    _tracker_metric_dict = []
    # --------------------------------------------------------------------------
    
    f = h5py.File(config["data_config"]["Processed_data_path"], "r")

    x_train = input_q(f['Background_data']['Train']['DATA'][:])
    x_test = input_q(f['Background_data']['Test']['DATA'][:])


    x_train_background = np.reshape(x_train, (x_train.shape[0], -1))
    x_test_background = np.reshape(x_test, (x_test.shape[0], -1))
    
    total_num_features = x_test_background.shape[-1]

    scale = f['Normalisation']['norm_scale'][:]
    bias = f['Normalisation']['norm_bias'][:]

    l1_bits_bkg_test = f['Background_data']['Test']['L1bits'][:]

    # --------------------------------------------------------------------------

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


    optim_config_contrastive = config["train"]["Contrastive_VAE"]["optimiser_config_contrastive"]
    optim_name_contrastive = optim_config_contrastive['optmiser']
    
    try:
        compute_optim = getattr(axo.optim, optim_name_contrastive)
    except AttributeError:
        print("Optimizer Not found in AXO, looking in TF")
        compute_optim = getattr(tf.keras.optimizers, optim_name_contrastive)
    
    allowed_params_for_optim = utilities.allowed_params(compute_optim)
    allowed_keys = list(set(optim_config_contrastive.keys()).intersection(set(allowed_params_for_optim)))
    config_to_optim_contrastive = {k: v for k, v in optim_config_contrastive.items() if k in allowed_keys}
    opt_contrastive = compute_optim(**config_to_optim_contrastive)


    model.compile(optimizer=opt_contrastive)

    _scheduler_flag = True
    if config["lr_schedule"]["Contrastive_VAE"]["Contrastive_stage"]["type"] != "CDW":
        print("'[WARNING] VICReg requires Cosine decay to train with stability, switching to constant LR decay mode not found")
        _scheduler_flag = False
    else:
        scheduler = cosine_with_warmup(
        max_lr=config["train"]["Contrastive_VAE"]["optimiser_config_contrastive"]["learning_rate"],
        warmup_epochs=config["lr_schedule"]["Contrastive_VAE"]["Contrastive_stage"]["config"]["warmup_epochs"],
        decay_epochs=Epochs_contrastive - config["lr_schedule"]["Contrastive_VAE"]["Contrastive_stage"]["config"]["warmup_epochs"])

    if DEBUG:
        print("DEBUG Flag found, training contrastive stage for 1 epoch only")
        Epochs_contrastive = 1

    for present_epoch in tqdm(range(0, Epochs_contrastive, 1)):
    
        index = torch.randperm(dataset.shape[0])
        if _scheduler_flag:
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

        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        metric_embed = {}
        metric_embed['TrainLossC'] = epoch_loss
        metric_embed['EpochC'] = present_epoch
        metric_embed['LrC'] = current_lr
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        
        _tracker_total_loss_vicreg.append(epoch_loss)
        _tracker_repr_loss_vicreg.append(epoch_repr)
        _tracker_std_loss_vicreg.append(epoch_std)
        _tracker_std_cov_vicreg.append(epoch_cov)
        _tracker_epoch_contrastive.append(present_epoch)
        _tracker_lr_contrastive.append(current_lr)

        logging.info(
        f"Epoch: {present_epoch}, LR: {current_lr}, "
        f"Total Loss: {epoch_loss:.4f}, Repr Loss: {epoch_repr:.4f}, "
        f"Std Loss: {epoch_std:.4f}, Cov Loss: {epoch_cov:.4f}"
        )
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

    
    model = VariationalAutoEncoder(encoder=encoder, decoder=decoder, kl_scale=kl_scale, reco_scale=reco_scale)


    optim_config_vae = config["train"]["Contrastive_VAE"]["optimiser_config_vae"]
    optim_name_vae = optim_config_vae['optmiser']
    
    try:
        compute_optim = getattr(axo.optim, optim_name_vae)
    except AttributeError:
        print("Optimizer Not found in AXO, looking in TF")
        compute_optim = getattr(tf.keras.optimizers, optim_name_vae)
    
    allowed_params_for_optim = utilities.allowed_params(compute_optim)
    allowed_keys = list(set(optim_config_vae.keys()).intersection(set(allowed_params_for_optim)))
    config_to_optim_vae = {k: v for k, v in optim_config_vae.items() if k in allowed_keys}
    opt_vae = compute_optim(**config_to_optim_vae)
    
    
    model.compile(optimizer=opt_vae)


    _scheduler_flag = True
    try:
        lrsc_config = config["lr_schedule"]["Contrastive_VAE"]["VAE_stage"]
    except KeyError:
        print("Learning rate scheduler not found; training with constant loss")
        _scheduler_flag = False
    else:
        lrsc_name = lrsc_config["type"]
        lrsc_config = lrsc_config["config"]
        lrsc_config["max_lr"] = config["train"]["Contrastive_VAE"]["optimiser_config_vae"]["learning_rate"]
        try:
            compute_lrsc = getattr(axo.utilities.lr_schedulers, lrsc_name)
        except AttributeError:
            print("Scheduler Not found in AXO, looking in TF callbacks")
            compute_lrsc = getattr(tf.keras.callbacks, lrsc_name)

    
        # Implementing the lrsc
        allowed_params_for_lrsc = utilities.allowed_params(compute_lrsc)
        allowed_keys = list(set(lrsc_config.keys()).intersection(set(allowed_params_for_lrsc)))
        config_to_lrsc = {k: v for k, v in lrsc_config.items() if k in allowed_keys}
        scheduler = compute_lrsc(**config_to_lrsc)
    
    
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

    if DEBUG:
        print("DEBUG Flag found, training VAE stage for 1 epoch only")
        Epochs_VAE = 1
    
    for present_epoch in tqdm(range(0, Epochs_VAE, 1)):
    
        index = torch.randperm(dataset_latent.shape[0]).numpy()
        
        if _scheduler_flag:
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
        
        _tracker_epoch_vae.append(present_epoch)
        _tracker_lr_vae.append(current_lr)
        _tracker_total_vae_loss.append(total_loss)
        _tracker_reco_loss_vae.append(reco_loss)
        _tracker_kld_loss_vae.append(kl_loss)
        _tracker_metric_dict.append(metric)

        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        metric['EpochVae'] = present_epoch
        metric['LrVae'] = current_lr
        metric['TotalLossVae'] = total_loss
        metric['RecoLossVae'] = reco_loss
        metric['KLLossVae'] = kl_loss
        if ray_backend:
            ray.train.report(metrics=metric) 
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        logging.info(
        f"Epoch: {present_epoch}, LR: {current_lr}, "
        f"Total Loss: {total_loss:.4f}, Reco Loss: {reco_loss:.4f}, "
        f"KL Loss: {kl_loss:.4f}"
        )

    #####################################################################
    # Preparing the history
    #####################################################################
    history["total_loss_vicreg"] = _tracker_total_loss_vicreg
    history["repr_loss_vicreg"] = _tracker_repr_loss_vicreg
    history["std_loss_vicreg"] = _tracker_std_loss_vicreg
    history["std_cov_vicreg"] = _tracker_std_cov_vicreg
    history["epoch_contrastive"] = _tracker_epoch_contrastive
    history["lr_contrastive"] = _tracker_lr_contrastive
    
    history["epoch_vae"] = _tracker_epoch_vae
    history["lr_vae"] = _tracker_lr_vae
    history["reco_loss_vae"] = _tracker_reco_loss_vae
    history["kld_loss_vae"] = _tracker_kld_loss_vae
    history["total_vae_loss"] = _tracker_total_vae_loss

    # Straightning the metric dictionary
    _flat_metric = defaultdict(list)
    
    for d in _tracker_metric_dict:
        for key, value in d.items():
            _flat_metric[key].append(value)

    history.update(dict(_flat_metric))
    #####################################################################
    # Merging the model elements
    #####################################################################

    input_layer = tf.keras.layers.Input(shape=(total_num_features,))
    embedding = input_layer
    
    for layer in vic_encoder.model.layers:
        embedding = layer(embedding)
    
    reco = embedding
    for layer in model.encoder.model.layers:
        reco = layer(reco)

    reco = model.encoder.layer_mu(reco)
    
    for layer in model.decoder.model.layers:
        reco = layer(reco)

    score = tf.keras.layers.Subtract()([reco,embedding])
    score = tf.keras.layers.Dot(axes=(1))([score,score])
    complete_model = tf.keras.Model(inputs=input_layer, outputs=score)

    gc.collect()
    if not ray_backend:
        return complete_model,vic_encoder,model, history