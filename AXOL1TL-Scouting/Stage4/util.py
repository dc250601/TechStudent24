import numpy as np
import json
import os
from collections import defaultdict
import time

def read_sparse(data, nbins):
    dense = np.zeros(nbins)
    for idx, val in data.items():
        dense[int(idx)] = val
    return dense

def get_meta(file_path):
    file = open(file_path,"r")
    file_data = file.readlines()
    file.close()
    meta_data = json.loads(file_data[1])
    lumi = int(meta_data["luminosity_blocks"][0])
    run = int(meta_data["run_numbers"][0])
    orbits = meta_data["orbit_numbers"]
    return {"lumisection":lumi,"run":run,"orbits":orbits}

def get_data(file_path):
    file = open(file_path,"r")
    file_data = file.readlines()
    file.close()

    data = json.loads(file_data[0]) 
    counts = data["counts"]
    bins = data["bins"]
    LBIN,UBIN,BINW = bins
    NBIN = (UBIN-LBIN)//BINW

    data = read_sparse(data["counts"],NBIN)

    return data,bins


def populate_hash(files,hash_map):
    for file in files:
        try:
            meta = get_meta(file)
            hash_map[meta["run"]][meta["lumisection"]].append(file)
        except:
            pass
    return hash_map

def check_duplicate_pair(meta1,meta2):

    assert meta1["run"] == meta2["run"], "Run numbers do not match!!!"
    assert meta1["lumisection"] == meta2["lumisection"], "Lumisection numbers do not match!!!"

    orb1 = set(meta1["orbits"])
    orb2 = set(meta2["orbits"])
    intersect = len(orb1.intersection(orb2))
    union = len(orb1.union(orb2))

    duplication_level = intersect/union

    if duplication_level < 0.1:
        return False
    else:
        return True

def check_duplicate(meta_list):
    ## I can write this in O(n) but won't as this will be too difficult to read.
    ## The complexity is O(n^2), I know you are smart but leave this readable please :)
      
    for i,meta1 in enumerate(meta_list):
        for j,meta2 in enumerate(meta_list):
            if i!=j:
                if check_duplicate_pair(meta1,meta2):
                    return True
                else:
                    continue
            else:
                break
    
    return False

def remove_duplicate(file_list):
    duplicate_files = []
    nfiles = len(file_list)
    for i in range(nfiles):
        for j in range(nfiles):
            if i>j:
                meta_i = get_meta(file_list[i])
                meta_j = get_meta(file_list[j])
                if check_duplicate_pair(meta_i,meta_j):
                    duplicate_files.extend([file_list[i],file_list[j]])
            else:
                break

    duplicate_files = list(set(duplicate_files))[:-1]

    for files in duplicate_files:
        os.remove(files)
        file_list.remove(files)
    return file_list



def sparse_dict_maker(l):
    s_dict = {int(i): float(v) for i, v in enumerate(l) if v != 0}
    return s_dict


def process_shards(hash_shards,
                   N_SHARDS,
                   save_dir
                    ):
    modified = False
    while(True):
        for run in hash_shards.keys():
            for lumi in hash_shards[run].keys():
                
                shard_list = hash_shards[run][lumi]

                if len(shard_list) == N_SHARDS:
                    print(f"Shard match for run {run} lumisection {lumi}")
                    meta_list = list(map(lambda x:get_meta(x), hash_shards[run][lumi]))

                    if check_duplicate(meta_list) == False:
                        print(f"Shards for run {run} lumisection {lumi} are not duplicates, moving to merge")
                        hash_shards = merge_and_forget(hash_shards, run,lumi,save_dir)
                        modified = True
                        break
                    else:
                        print(f"Shards for run {run} lumisection {lumi} are duplicates, removing duplicates")
                        hash_shards[run][lumi] = remove_duplicate(shard_list)
                        modified = True
                        break
                
                elif len(shard_list) > N_SHARDS:
                    print(f"Shard overflow for run {run} lumisection {lumi}")
                    hash_shards[run][lumi] = remove_duplicate(shard_list)
                    modified = True

                if modified:
                    break
            if modified:
                modified = False
                break
            else:
                return 0
    

def merge_and_forget(hash_map,run,lumi,save_dir):

    file_list = hash_map[run][lumi]

    last_bins = None
    all_counts = None

    print("Merging shards for run (inside merge_and_forget)",run,"lumi",lumi)

    for file in file_list:
        try:
            counts,bins = get_data(file)
            if last_bins:
                if last_bins != bins:
                    print("Warning Binnings do not match, results will be anomalous!!!!")
            
            if type(all_counts) != type(None):
                all_counts += counts
            else:
                all_counts = counts
            last_bin = bins

        except :
            print("Something went wrong merging the shards the files,Will retry later!!!")
            # return 1

    sparse_counts = sparse_dict_maker(all_counts)

    # Deleting hash key
    del hash_map[run][lumi]

    # Deleting the shards
    for file in file_list:
        os.remove(file)

    file_name = f"run_{run}_lumi_{lumi}.axo"
    file_name = os.path.join(save_dir,file_name)

    print("saving-file",file_name)
    output_file = open(file_name,"w")

    print("Saving histogram and metadata to",file_name)

    meta = {"run_numbers":run,
            "luminosity_blocks":lumi,
            }

    meta = json.dumps(meta, default=str)

    histogram = {
        "bins":bins,
        "counts":sparse_counts}
    histogram = json.dumps(histogram, default=str)

    output_file.write(histogram + "\n")
    output_file.write(meta + "\n")
    utc_now = time.gmtime()
    utc_now = time.strftime("%Y-%m-%d %H:%M:%S", utc_now)

    output_file.write(f"Current time UTC: {utc_now}")
    output_file.close()

    return hash_map


