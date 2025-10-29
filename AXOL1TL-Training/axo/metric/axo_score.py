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

class axo_threshold_manager():
    def __init__(self,model,config):
        self.config = config
        self.model = model
        
        self.target_rate = self.config["target_rate"]
        self.threshold_raw = None
        self.threshold_pure = None
        self.bc_rate_khz = self.config["bc_khz"]
        
        self.score_dict = {} #Internal variable not to be accessed directly
        self.data_file = None #Internal variable not to be accessed directly
        
        self.data_path = self.config["data_path"]
        self.HT_THRESHOLD = self.config["ht_threshold"]
        self.bkg_score = None
        self.bkg_L1 = None

        self.signal_score = {}
        self.signal_L1 = {}
        self.signal_HT = {}
        self.raw_rate_wrt_pure = {}
        self.pu_values = None
        self.total_rates_khz = None
        self.rate_uncertainties = None

        self.sn_score = None
        self.zb_score = None
        
        # Init ...
        self._open_hdf5()
        
        self.get_bkg_scores()
        self.get_signal_scores()
        
        self.get_threshold_raw()
        self.get_threshold_pure()
        
        self.get_axo_efficiencies()
        self.get_raw_wrt_pure()

        
    def _open_hdf5(self):
        self.data_file = h5py.File(self.data_path,"r")

    def get_bkg_scores(self):
        x_test = self.data_file["Background_data"]["Test"]["DATA"][:]
        x_test = np.reshape(x_test,(x_test.shape[0],-1))       
        self.bkg_score = self.model.predict(x_test,batch_size = 120000)
        self.zb_score = self.bkg_score
        self.bkg_L1 = self.data_file["Background_data"]["Test"]["L1bits"][:]
        del x_test
        gc.collect()
    
    def get_threshold_single_nu(self):
        x_test = self.data_file["Signal_data"]["SingleNeutrino_E-10-gun"]["DATA"][:]
        x_test = np.reshape(x_test, (x_test.shape[0], -1))       
        pu_data = np.array(self.data_file["Signal_data"]["SingleNeutrino_E-10-gun"]["PU"])
        latent_axo_qk = self.model.predict(x_test, batch_size=120000)
        y_axo_qk = np.sum(latent_axo_qk**2, axis=1)

        self.sn_score = y_axo_qk
        self.pu_values = {}
        self.total_rates_khz = {}
        self.rate_uncertainties = {}  
        
    
        pu_bins = np.arange(0, 71, 5) 
        pu_centers = (pu_bins[:-1] + pu_bins[1:]) / 2

        for thres_key, thres_value in self.threshold_pure.items():
            rates_for_pu = []
            rate_errors = [] 
            
            for i in range(len(pu_bins)-1):
                pu_mask = (pu_data >= pu_bins[i]) & (pu_data < pu_bins[i+1])
                events_in_bin = np.sum(pu_mask)
                
                if events_in_bin > 0:
                    y_axo_qk_pu = y_axo_qk[pu_mask]
                    
                    triggered_events = np.sum(y_axo_qk_pu > thres_value)
                    
                    bc_rate_per_bin = self.bc_rate_khz / (len(pu_bins) - 1)
                    rate = (triggered_events / events_in_bin) * bc_rate_per_bin
                    
                    p = triggered_events / events_in_bin
                    uncertainty = np.sqrt((p * (1 - p)) / events_in_bin) * bc_rate_per_bin
                    
                    rates_for_pu.append(rate)
                    rate_errors.append(uncertainty)
                else:
                    rates_for_pu.append(0)
                    rate_errors.append(0)
            

            self.pu_values[thres_key] = pu_centers
            self.total_rates_khz[thres_key] = np.array(rates_for_pu)
            self.rate_uncertainties[thres_key] = np.array(rate_errors)

    def get_signal_scores(self):
        signal_names = self.data_file["Signal_data"].keys()
        for s in signal_names:
            
            sig_data = self.data_file["Signal_data"][s]["DATA"][:]
            sig_data = np.reshape(sig_data,(sig_data.shape[0],-1))
            
            self.signal_score[s] = self.model.predict(sig_data,
                                                      batch_size = 120000,
                                                      verbose=0)
            self.signal_L1[s] = self.data_file["Signal_data"][s]["L1bits"][:]
            self.signal_HT[s] = self.data_file["Signal_data"][s]["HT"][:]
        del sig_data
        gc.collect()
        
    def get_threshold_raw(self):
            
        threshold = {}
        for target_rate in self.target_rate:
            threshold[target_rate] = np.percentile(self.bkg_score, 100-(target_rate/self.bc_rate_khz)*100)
        self.threshold_raw = threshold.copy()

    def get_threshold_pure(self):
        threshold = {}
        rates = 10**np.arange(-2,2.1,0.1)
        
        for rate in rates:
            threshold[rate] = np.percentile(self.bkg_score, 100 - (rate / self.bc_rate_khz) * 100)
            
        raw_rate = []
        pure_rate = []
        for thres in threshold.keys():
            nsamples = self.bkg_score.shape[0]
            axo_triggered = np.where(self.bkg_score > threshold[thres])[0]
            l1_triggered = np.where(self.bkg_L1)[0]
            pure_triggered = np.setdiff1d(axo_triggered, l1_triggered)
            raw_rate.append((axo_triggered.shape[0] * self.bc_rate_khz) / nsamples)
            pure_rate.append((pure_triggered.shape[0] * self.bc_rate_khz) / nsamples)
        
        threshold_pure = {}
        for thres in self.target_rate:
            _pure_rate = thres
            _raw_rate = np.interp(_pure_rate, xp=pure_rate, fp=raw_rate)
            threshold_pure[thres] = np.percentile(self.bkg_score, 100 - (_raw_rate / self.bc_rate_khz) * 100)        
        self.threshold_pure = threshold_pure.copy()
        
    def get_axo_efficiencies(self):
        
        HT_THRESHOLD = self.HT_THRESHOLD
        signal_names = self.signal_score.keys()
        
        score = {}
        score["SIGNAL_NAMES"] = signal_names
        score["SCORE"] = {}
        
        for rate in self.target_rate:
            _l1_eff = []
            
            _raw_raw_eff = []
            _pure_raw_eff = []
            _raw_pure_eff = []
            _pure_pure_eff = []
            
            _ht_eff = []
            
            for signal in signal_names:
                nsamples = self.signal_score[signal].shape[0]

                # Calculating for the raw rates
                raw_triggered = np.where(self.signal_score[signal] > self.threshold_raw[rate])[0]
                pure_triggered = np.setdiff1d(ar1=raw_triggered, ar2=np.where(self.signal_L1[signal])[0])

                _raw_raw_eff.append(100*raw_triggered.shape[0]/nsamples)
                _pure_raw_eff.append(100*pure_triggered.shape[0]/nsamples)

                # Calculating for the pure rates
                raw_triggered = np.where(self.signal_score[signal] > self.threshold_pure[rate])[0]
                pure_triggered = np.setdiff1d(ar1=raw_triggered, ar2=np.where(self.signal_L1[signal])[0])                

                _raw_pure_eff.append(100*raw_triggered.shape[0]/nsamples)
                _pure_pure_eff.append(100*pure_triggered.shape[0]/nsamples)

                # HT and L1
                l1_triggered = np.where(self.signal_L1[signal])[0]
                ht_triggered = np.where(self.signal_HT[signal] > HT_THRESHOLD)[0]
                
                _l1_eff.append(100*l1_triggered.shape[0]/nsamples)
                _ht_eff.append(100*ht_triggered.shape[0]/nsamples)
                
                
            score["SCORE"][rate] = {
                "raw_raw-axo":_raw_raw_eff,
                "pure_raw-axo":_pure_raw_eff,
                "raw_pure-axo":_raw_pure_eff,
                "pure_pure-axo":_pure_pure_eff,
                "L1_eff":_l1_eff,
                "HT_eff":_ht_eff,
            }

        self.score_dict = score ## Storing it here
        
    def get_raw_dict(self):
        return self.score_dict,self.threshold

    def get_raw_wrt_pure(self):
        
        for rate in self.target_rate:
            number_of_events = np.where(self.bkg_score>self.threshold_pure[rate])[0].shape[0]
            rate_ = (number_of_events/self.bkg_score.shape[0])*self.bc_rate_khz
            self.raw_rate_wrt_pure[rate] = rate_
       
    def get_score(self,thres):
                
        signal_names = self.signal_score.keys()
        df = pd.DataFrame()
        
        df["Signal Name"] = signal_names
        
        df["AXO raw-raw"] = self.score_dict["SCORE"][thres]['raw_raw-axo']
        df["AXO pure-raw"] = self.score_dict["SCORE"][thres]['pure_raw-axo']
        
        df["AXO raw-pure"] = self.score_dict["SCORE"][thres]['raw_pure-axo']
        df["AXO pure-pure"] = self.score_dict["SCORE"][thres]['pure_pure-axo']
        
        df["L1 Efficiency"] = self.score_dict["SCORE"][thres]['L1_eff']
        df["HT Efficiency"] = self.score_dict["SCORE"][thres]['HT_eff']
        
        return df