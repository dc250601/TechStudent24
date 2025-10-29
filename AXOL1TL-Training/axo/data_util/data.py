from . import _normalise
from . import _pack
from . import _read
from . import _saturation
import gc

def get_data(config_master):
    # This config is supposed to be the data config for the file
    
    ### Reading the data
    config_bkg = config_master["Read_configs"]["BACKGROUND"]
    config_sig = config_master["Read_configs"]["SIGNAL"]
    
    data_bkg = _read.get_data_background(config=config_bkg,
                                         object_ranges = config_master["Read_configs"]["object_ranges"],
                                         constituents = config_master["Read_configs"]["constituents"],
                                        )
    data_sig = _read.get_data_signal(config=config_sig,
                                     object_ranges = config_master["Read_configs"]["object_ranges"],
                                     constituents = config_master["Read_configs"]["constituents"],
                                    )
    #############################################################################
    ### Saturation Treatment
    saturation_mode = config_master["Saturation_configs"]["saturation_mode"]

    data_bkg = _saturation._remove_saturation(data_bkg,
                                              config_master["Read_configs"]["constituents"],
                                              saturation_mode=saturation_mode)
    data_sig = _saturation._remove_saturation(data_sig,
                                              config_master["Read_configs"]["constituents"],
                                              saturation_mode=saturation_mode)
    #############################################################################
    ### Normalisation
    scheme = config_master["Normalisation_configs"]["scheme"]
    norm_ignore_zeros = config_master["Normalisation_configs"]["norm_ignore_zeros"]

    data_bkg, data_sig, norm_bias, norm_scale = _normalise(data_bkg = data_bkg,
                                                           data_sig = data_sig,
                                                           scheme = scheme,
                                                           norm_ignore_zeros = norm_ignore_zeros)
    #############################################################################
    ### Packing and storing
    _pack(data_bkg = data_bkg,
                data_sig=data_sig,
                config=config_master,
                norm_bias=norm_bias,
                norm_scale=norm_scale
               )
    
    
    del data_bkg, data_sig
    gc.collect()