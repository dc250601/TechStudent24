from util import worker_init, evaluate_and_store
import multiprocessing as mp
import zarr
import os
import glob
from tqdm.auto import tqdm
import gc
import time


if __name__ == "__main__":
    
    stage1_dir = "/dev/shm/StoreStage1/"
    save_dir = "/dev/shm/StoreStage2/"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    mp.set_start_method("spawn", force=True)
    
    NUM_THREADS = 16
    pool = mp.Pool(NUM_THREADS, initializer=worker_init, initargs=(save_dir,))
    

    while(True):
        accessible_files = glob.glob(os.path.join(stage1_dir,"*.zarr"))
        readable_files = []
        for file in tqdm(accessible_files):
            f = zarr.open_group(file,"r")
            try:
                if "WriteTime" in f.attrs:
                    readable_files.append(file)
            except:
                continue

        if len(readable_files) == 0:
            time.sleep(1)
            continue
        # else:
        for i in tqdm(range(0,len(readable_files),NUM_THREADS)):
            result = list(pool.imap(evaluate_and_store,readable_files[i:i+NUM_THREADS]))
            gc.collect()

    pool.close()