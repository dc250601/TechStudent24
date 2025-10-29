import json
import gc
import numpy
import os
import h5py

def _pack(data_bkg,data_sig,norm_bias, norm_scale,config):
    path = config["Processed_data_path"]
    
    compression_mode = config["Write_compression"]["Compression_mode"]
    compression_opts=config["Write_compression"]["Compression_opts"]

    if not compression_mode:
        compression_mode = "gzip"
        compression_opts = 0
        
    if os.path.exists(path):
        os.remove(path)
    
    f = h5py.File(path,"w")
    
    
    # Adding the configs ....
    config = json.dumps(config)
    f.attrs["config"] = config
    # Filling the Background data
    
    f.create_group("Background_data")
    for event in data_bkg.keys():
        f["Background_data"].create_group(event)
        # Filling the Meta data first
        for meta_key in data_bkg[event]["META"].keys():
            
            f["Background_data"][event].create_dataset(name=meta_key,
                                                       data=data_bkg[event]["META"][meta_key],
                                                       compression=compression_mode,
                                                       compression_opts=compression_opts
                                                      )
        # Filling the data
        f["Background_data"][event].create_dataset(name="DATA",
                                                   data=data_bkg[event]["DATA"],
                                                   compression=compression_mode,
                                                   compression_opts=compression_opts
                                                  )
    # Filling the signal data
    f.create_group("Signal_data")
    for event in data_sig.keys():
        f["Signal_data"].create_group(event)
        # Filling the Meta data first
        for meta_key in data_sig[event]["META"].keys():
            f["Signal_data"][event].create_dataset(name=meta_key,
                                                   data=data_sig[event]["META"][meta_key],
                                                   compression=compression_mode,
                                                   compression_opts=compression_opts

                                                  )
        # Filling the data
        f["Signal_data"][event].create_dataset(name="DATA",
                                               data=data_sig[event]["DATA"],
                                               compression=compression_mode,
                                               compression_opts=compression_opts
                                              )
    # Filling the norms and biases
    f.create_group("Normalisation")
    f["Normalisation"].create_dataset(
        name = "norm_scale",
        data = norm_scale,
        compression=compression_mode,
        compression_opts=compression_opts

    )
    
    f["Normalisation"].create_dataset(
        name = "norm_bias",
        data = norm_bias,
        compression=compression_mode,
        compression_opts=compression_opts

    )

    f.close()
    gc.collect()
    
    