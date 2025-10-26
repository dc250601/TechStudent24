import os
import glob
import uproot
from util import get_events_hw_flat
import traceback
import time

import multiprocessing
if __name__ == "__main__":

    multiprocessing.set_start_method("forkserver")
    
    file_location ="/dev/shm/StoreStage0/"
    store_dir = "/dev/shm/StoreStage1"

    if not os.path.exists(store_dir):
        os.makedirs(store_dir)

    NUM_THREADS = 32
    pool = multiprocessing.Pool(NUM_THREADS)
    
    while(True):
        files = glob.glob(os.path.join(file_location,"*.root"))
        if len(files) == 0:
            time.sleep(1)
        else:
            for file_path in files:
                try: ## Sometime due to race conditions the reading fails...
                    f = uproot.open(file_path)
                    NUM_ENTRIES = f["Events"].num_entries
                    f.close()
                    print("Sending to starmap",file_path)
                    BATCH_SIZE = NUM_ENTRIES // NUM_THREADS + 1
                    results = pool.starmap(get_events_hw_flat, [(file_path, i*BATCH_SIZE, (i+1)*BATCH_SIZE,store_dir) for i in range(NUM_THREADS)])
                    os.remove(file_path)
                except Exception as e:
                    print(f"Something went wrong with {file_path}: {e!r}")
                    traceback.print_exc()    



    pool.close()