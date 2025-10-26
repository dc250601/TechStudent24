import multiprocessing
mp = multiprocessing
import zarr
import os
import glob
import numpy as np
from tqdm.auto import tqdm
import h5py as h5
import gc
 
import json
import time
import random
import shutil

complete_model = None
save_dir = "none"
tf_global = None

def worker_init(save_dir_arg):
    global complete_model
    global save_dir
    save_dir = save_dir_arg
    global tf_global
    
    import tensorflow as tf
    tf_global = tf

    physical_devices = tf.config.list_physical_devices('GPU')
    for dev in physical_devices:
        tf.config.experimental.set_memory_growth(dev, True)
    from qkeras import quantized_bits
    from qkeras.utils import _add_supported_quantized_objects

    co = {}
    _add_supported_quantized_objects(co)
    base_model = tf.keras.models.load_model('/eos/project/c/cms-axol1tl/Pipeline/Models/Deployed/build.h5', custom_objects=co) # DO NOT HACK THE PATHS !!!

    with h5.File("/eos/project/c/cms-axol1tl/Pipeline/Models/Deployed/complete.h5", "r") as f:  # DO NOT HACK THE PATHS !!!
        scale = np.reshape(f['data']["Normalisation"]["norm_scale"][:], (57,))
        bias  = np.reshape(f['data']["Normalisation"]["norm_bias"][:], (57,))
        bits  = json.loads(f.attrs["config"])["data_config"]['Quantization_configs']['quantize_bits']
    
    # quantizer = quantized_bits(*bits, alpha=1)
    # inp = tf.keras.Input(57)
    # x = tf.keras.layers.subtract([inp, bias[None, :]])
    # x = tf.keras.layers.Multiply()([x, 1/scale[None, :]])
    # x = quantizer(x)
    # x = base_model(x)
    # complete_model = tf.keras.Model(inp, x) # This is the correct way to do it !!!!

    input_quantiser = quantized_bits(14,5, alpha=1)
    output_quantizer = quantized_bits(18,13,alpha=1)
    bias_quantizer = quantized_bits(18,12,alpha=1)
    inp = tf.keras.Input(57)
    x = tf.keras.layers.subtract([inp, bias[None, :]])
    x = bias_quantizer(x)
    x = tf.keras.layers.Multiply()([x, 1/scale[None, :]])
    x = bias_quantizer(x)
    x = input_quantiser(x)
    x = base_model(x)
    x = output_quantizer(x)
    complete_model = tf.keras.Model(inp, x) ## This matches the Firmware.





def evaluate_and_store(file_path):
    global complete_model
    global save_dir
    
    file = zarr.open_group(file_path, "r")
    data = file["Data"][:]

    
    # axo_score = complete_model.predict(data,batch_size = 2**17,verbose=0) ## This might cause OOM errors.
    ## This is optimized for A100s and H100. Atleast 30GB of VRAM is required. Adjust otherwise.
    
    batch_size = 2**19
    outputs = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        out = complete_model(batch, training=False)
        outputs.append(out)

    axo_score = tf_global.concat(outputs, axis=0).numpy()

    file_name = str(time.time_ns()) 
    file_name = file_name + "_" + str(int(random.random()*10000))
    file_name = file_name + "_" + str(int(random.random()*10000))
    file_name = file_name + ".zarr"
    
    file_name = os.path.join(save_dir,file_name)
    
    file_write = zarr.open_group(file_name,
                           mode="w",
                           zarr_version=2)
    
    file_write["AxoScore"] = axo_score
    file_write["RunNumber"] = file["RunNumber"]
    file_write["LuminosityBlock"] = file["LuminosityBlock"]
    file_write["BunchCrossing"] = file["BunchCrossing"]
    file_write["OrbitNumber"] = file["OrbitNumber"]
    file_write["nl1EG"] = file["nl1EG"]
    file_write["nL1Jet"] = file["nL1Jet"]
    
    file_write.attrs["WriteTime"] = time.time_ns() ## Acts like a lock !
    file_write.store.close()

    shutil.rmtree(file_path)    
    del data, axo_score
    gc.collect()
    return 0


