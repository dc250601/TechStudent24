from util import worker_init, create_and_store_hist
import multiprocessing as mp
import zarr
import os
import glob
from tqdm.auto import tqdm
import gc
import time

if __name__ == "__main__":
    
    stage2_dir = "/dev/shm/StoreStage2/"
    save_dir = "/dev/shm/StoreStage3/"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    mp.set_start_method("spawn", force=True)
    
    NUM_THREADS = 8
    pool = mp.Pool(NUM_THREADS, initializer=worker_init, initargs=(save_dir,))
    

    while(True):
        accessible_files = glob.glob(os.path.join(stage2_dir,"*.zarr"))
        readable_files = []
        for file in tqdm(accessible_files):
            f = zarr.open_group(file,"r")
            if "WriteTime" in f.attrs:
                readable_files.append(file)
                
        if len(readable_files) == 0:
            time.sleep(2)
            continue
        
        for i in tqdm(range(0,len(readable_files),NUM_THREADS)):
            result = list(pool.imap(create_and_store_hist,readable_files[i:i+NUM_THREADS]))
            gc.collect()

    pool.close()