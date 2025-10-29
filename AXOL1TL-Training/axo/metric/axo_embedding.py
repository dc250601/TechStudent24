import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import json

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Input
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf

import gc

class axo_embedding():
    def __init__(self,model,enc,vae,config,axo_manager):
        self.config = config
        self.model = model
        self.axo_man = axo_manager

        self.enc = enc
        self.vae = vae
        
        self.test_bkg_embedding_vicreg = None
        self.signal_embedding_dict_vicreg = None

        self.test_bkg_embedding_vae = None
        self.signal_embedding_dict_vae = None

        self.test_bkg_embedding_vae_reco = None
        self.signal_embedding_dict_vae_reco = None

        self.get_embedding()
        
        
    def get_embedding(self):
        #######################################################
        test_bkg = self.axo_man.data_file["Background_data"]["Test"]["DATA"][:]
        test_bkg = test_bkg.reshape(-1,57)
        
        SIGNAL_NAMES = list(self.axo_man.data_file['Signal_data'].keys())
        signal_data_dict = {}
        
        for signal_name in SIGNAL_NAMES:
            x_signal = self.axo_man.data_file['Signal_data'][signal_name]['DATA'][:]
            x_signal = np.reshape(x_signal, (x_signal.shape[0], -1))
            signal_data_dict[signal_name] = x_signal
        #######################################################
        
        
        self.test_bkg_embedding_vicreg = self.enc.predict(test_bkg,batch_size=16000)
        
        self.signal_embedding_dict_vicreg = {}
        
        for signal_name in SIGNAL_NAMES:
            self.signal_embedding_dict_vicreg[signal_name] = self.enc.predict(signal_data_dict[signal_name],batch_size=16000)
        #######################################################

        
        self.test_bkg_embedding_vae,_ = self.vae.encoder.predict(self.test_bkg_embedding_vicreg,batch_size = 16000)
        self.signal_embedding_dict_vae = {}

        for signal_name in SIGNAL_NAMES:
            self.signal_embedding_dict_vae[signal_name],_ = self.vae.encoder.predict(self.signal_embedding_dict_vicreg[signal_name],batch_size=16000)
        #######################################################

        
        self.test_bkg_embedding_vae_reco = self.vae.decoder.predict(self.test_bkg_embedding_vae,batch_size = 16000)
        self.signal_embedding_dict_vae_reco = {}

        for signal_name in SIGNAL_NAMES:
            self.signal_embedding_dict_vae_reco[signal_name] = self.vae.decoder.predict(self.signal_embedding_dict_vae[signal_name],batch_size=16000)        
        #######################################################

        

        

    
        