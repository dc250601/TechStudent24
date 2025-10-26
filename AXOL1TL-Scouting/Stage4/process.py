import numpy as np
import os
import glob
import json
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import h5py as h5
from collections import defaultdict
import time
import sys
from util import process_shards, populate_hash

if __name__ == "__main__":
    
    save_dir = "/eos/project/c/cms-axol1tl/Pipeline/Scouting/ScoreStream"
    stage3_dir = "/dev/shm/StoreStage3"
    
    N_SHARDS = 32
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    while(True):
        if os.path.exists(stage3_dir):
            files = glob.glob(os.path.join(stage3_dir,"*.axo_shards"))
            print("Files to process",len(files))
            if len(files) > 0:
                hash_merge = defaultdict(lambda: defaultdict(list))
                hash_merge = populate_hash(files = files,hash_map= hash_merge)
                _ =  process_shards(hash_merge,N_SHARDS,save_dir)
        time.sleep(1)