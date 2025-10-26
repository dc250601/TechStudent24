import torch ### DO NOT REMOVE THIS !!!!!

import os
import tensorflow as tf

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

import gc
import argparse

from sklearn.mixture import GaussianMixture
from scipy.spatial import distance
import random
import gc
h5py=h5


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
    
    blur_p = config["blur_p"]
    blur_m = config["blur_m"]
    blur_s = config["blur_s"]
    
    mask_p = config["mask_p"]   


    VIC_lr = config["vic_lr"]
    

    Epochs_contrastive = config["vic_epochs"]
    GMM_components = config["gmm_ncomponents"]
    
    Batch_size = 4096
    
    CUT = 0.01
    
    device = f"cuda:0"
    
    vic_encoder_nodes = config["encoder_nodes"]
    projector_features = vic_encoder_nodes[-1]*4

    #--------------------------------------------------------------------------
    
    f = h5py.File("<redacted>","r")
    
    x_train = f["Background_data"]["Train"]["DATA"][:]
    x_test = f["Background_data"]["Test"]["DATA"][:]
    
    x_train_background = np.reshape(x_train,(x_train.shape[0],-1))
    x_test_background = np.reshape(x_test,(x_test.shape[0],-1))
    
    scale = f["Normalisation"]["norm_scale"][:]
    bias = f["Normalisation"]["norm_bias"][:]
    
    l1_bits_bkg_test = f["Background_data"]["Test"]["L1bits"][:]

    SIGNAL_NAMES = list(f["Signal_data"].keys())

    signal_data_dict = {}
    signal_embedding_dict = {}
    signal_l1_dict = {}
    
    for signal_name in SIGNAL_NAMES:
        x_signal = f["Signal_data"][signal_name]["DATA"][:]
        x_signal = np.reshape(x_signal,(x_signal.shape[0],-1))
        l1_bits = f["Signal_data"][signal_name]["L1bits"][:]
    
        signal_data_dict[signal_name] = x_signal
        signal_l1_dict[signal_name] = l1_bits
    
    f.close()
    #--------------------------------------------------------------------------
    
    feature_blur = FastFeatureBlur(p = blur_p,
                                   strength=blur_s,
                                   magnitude = blur_m,
                                   device=device)
    feature_blur_prime = FastFeatureBlur(p = blur_p,
                                         strength=blur_s,
                                         magnitude = blur_m,
                                         device=device)
    
    object_mask = FastObjectMask(p=mask_p,device=device)
    object_mask_prime = FastObjectMask(p=mask_p,device=device)
    
    lorentz_rot = FastLorentzRotation(p=0.5,norm_scale=scale, norm_bias=bias, device = device)
    lorentz_rot_prime = FastLorentzRotation(p=0.5,norm_scale=scale, norm_bias=bias, device = device)
    
    #--------------------------------------------------------------------------
    dataset = torch.tensor(x_train_background,dtype=torch.float32,device=device)
    dataset_test = torch.tensor(x_test_background,dtype=torch.float32,device=device)
    del x_train_background
    gc.collect()
    
    #--------------------------------------------------------------------------
        
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
    
        vic_encoder = model.backbone
        dataset_latent_test = vic_encoder(dataset_test.cpu().numpy())
        for signal_name in SIGNAL_NAMES:
            signal_embedding_dict[signal_name] = vic_encoder(signal_data_dict[signal_name])
        
        index = torch.randperm(dataset_latent_test.shape[0])
        
        model_gm = GaussianMixture(n_components=GMM_components,
                                   random_state=0).fit(dataset_latent_test.numpy()[index[:int(index.shape[0]*CUT)]])
        distance = DistanceLayer(model = model_gm,
                                 bits=32,
                                 integer=8,
                                 batch_size=8192) ## Bits to be further tuned !!!
        
        metric = fast_score(
                model = model_gm,
                data_bkg=dataset_latent_test,
                bkg_l1_bits=l1_bits_bkg_test,
                distance_func=distance,
                data_signal=signal_embedding_dict,
                signal_l1_bits=signal_l1_dict,
                evaluation_threshold=1,
                tf_backend = True
        )
        
    
        metric['TrainLossC'] = epoch_loss
        metric['EpochC'] = present_epoch
        metric['LrC'] = current_lr
    
        wandb.log(metric)
        ray.train.report(metrics = metric)    
    
    #########################################################################################################################
    #########################################################################################################################
    #########################################################################################################################
    
    


    


if __name__ == "__main__":

    ap_fixed_kernel = [6,2] ### To be further tuned !!!!
    ap_fixed_bias = [10,6] ### To be further tuned !!!!
    ap_fixed_act = [10,6] ### To be further tuned !!!!
    ap_fixed_data = [8,5] ### To be further tuned !!!!
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address", 
        type=str, 
        default=None)
    
    args = parser.parse_args()
    
    
    if args.address:
        ray.init(address=args.address)
    else:
        ray.init(address="auto")
        
 
    search_space = {
        "vic_lr": tune.loguniform(1e-5,1e-3),
    
        "blur_p": tune.uniform(0,1),
        "blur_m": tune.uniform(0,1),
        "blur_s": tune.uniform(0,1),
        "mask_p": tune.uniform(0,1),
            
        "encoder_nodes": tune.sample_from(lambda spec: [tune.randint(24, 32).sample(), tune.randint(8, 18).sample()]),

        "vic_epochs":tune.choice([10,25,50,100,250,500,1000]),
    
        "gmm_ncomponents":tune.choice([2,3,4,5,6,7])
        
    }
        
        
    
    optuna_search = OptunaSearch(
        metric="pure-pure/haa4b-ma15",
        mode="max",)

    scheduler = ASHAScheduler(
        metric="pure-pure/haa4b-ma15",
        mode="max",
        max_t=480,
        grace_period=32,
        reduction_factor=2,
    )
    
    analysis = tune.run(
        run,
        config=search_space,
        search_alg=optuna_search,
        storage_path="<redacted>",
        scheduler=scheduler,
        num_samples=1000, 
        resources_per_trial={"cpu": 4, "gpu": 1/4}
    )

