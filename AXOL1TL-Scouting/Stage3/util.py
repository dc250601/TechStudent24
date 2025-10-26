import numpy as np
import os
import zarr
import gc
import json
import time
import random
import shutil
save_dir = "none"

def sparse_dict_maker(l):
    s_dict = {int(i): float(v) for i, v in enumerate(l) if v != 0}
    return s_dict

def worker_init(save_dir_arg):
    global save_dir
    save_dir=save_dir_arg

def create_and_store_hist(file_path):

    global save_dir

    BIN_BEGIN = 0
    BIN_END = 20000
    BIN_WIDTH = 1
    bins = np.arange(BIN_BEGIN,BIN_END,BIN_WIDTH)

    file_name = str(time.time_ns()) 
    file_name = file_name + "_" + str(int(random.random()*10000))
    file_name = file_name + "_" + str(int(random.random()*10000))
    file_name = file_name + ".axo_shards"
    
    file_name = os.path.join(save_dir,file_name)

    
    try:
        file = zarr.open(file_path,"r")
    except:
        time.sleep(0.5)
        try:
           file = zarr.open(file_path,"r")
        except:
            print(f"Something went from for {file_path} in  Stage 3, will try later")
            return 1

    data = file["AxoScore"][:]
    hist,_ = np.histogram(a=data,bins=bins)
    sparse_hist = sparse_dict_maker(hist)

    run_numbers = list(set(file["RunNumber"][:]))
    luminosity_blocks = list(set(file["LuminosityBlock"][:]))
    orbit_numbers = list(set(file["OrbitNumber"][:]))


    meta = {"run_numbers":run_numbers,
            "luminosity_blocks":luminosity_blocks,
            "orbit_numbers":orbit_numbers
            }

    meta = json.dumps(meta, default=str)

    histogram = {
        "bins":[BIN_BEGIN,BIN_END,BIN_WIDTH],
        "counts":sparse_hist}
    histogram = json.dumps(histogram, default=str)
    output_file = open(file_name,"w")
    output_file.write(histogram + "\n")
    output_file.write(meta + "\n")

    output_file.close()

    shutil.rmtree(file_path)

    del data, hist, meta
    gc.collect()
    return 0
