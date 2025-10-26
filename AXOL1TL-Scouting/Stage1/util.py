import numpy as np
import os
from tqdm import tqdm

import uproot
import awkward as ak
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import zarr, time, random
import shutil

def awkward_to_numpy(ak_array, maxN, verbosity = 0):
    selected_arr = ak.fill_none( ak.pad_none( ak_array, maxN, clip=True, axis=-1), {"pt":0, "eta":0, "phi":0})
    np_arr = np.stack( (selected_arr.pt.to_numpy(), selected_arr.eta.to_numpy(), selected_arr.phi.to_numpy()), axis=2)
    return np_arr.reshape(np_arr.shape[0], np_arr.shape[1] * np_arr.shape[2])


## The below function will be later accelarated using DASK
def get_events_hw_flat(file_path,begin_index,end_index,save_dir):
    
    # print("Withing rhe get event function")
    
    NEG = 4
    NMU = 4
    NJET = 10
    NTHRESHOLDS = 4
    branch_dict = {
        'L1Mu': ["hwPt", "hwEtaAtVtx", "hwPhiAtVtx"],
        'L1EG': ["hwPt", "hwEta", "hwPhi"],
        'L1Jet': ["hwPt", "hwEta", "hwPhi"],
        'L1EtSum': ["hwPt", "hwPhi"]
    }
    
    branches = [f"{obj}_{var}" for obj, vars in branch_dict.items() for var in vars]
    # print("Opening Uproot file")

    decomp = ThreadPoolExecutor(max_workers=os.cpu_count())
    f = uproot.open(file_path,
                    use_threads = True,
                    decomp = ThreadPoolExecutor(max_workers=os.cpu_count()),
                    object_cache=None
                    
                   )
    ################################################################
    b_meta = [
          "run",
          "luminosityBlock",
          "bunchCrossing",
          "orbitNumber",
          "nL1EG",
          "nL1Jet"]

    events_meta_raw = f["Events;1"].arrays(
                b_meta,
                entry_start=begin_index, 
                entry_stop=end_index
            )
    
    RunNumber = np.array(events_meta_raw.run)
    LuminosityBlock = np.array(events_meta_raw.luminosityBlock)
    BunchCrossing = np.array(events_meta_raw.bunchCrossing)
    OrbitNumber = np.array(events_meta_raw.orbitNumber)
    nL1EG = np.array(events_meta_raw.nL1EG)
    nL1Jet = np.array(events_meta_raw.nL1Jet)
    ################################################################
    events_raw = f["Events"].arrays(
        branches,
        entry_start=begin_index, 
        entry_stop=end_index
    )
    f.close()
    # print("File closed")
    new_arr = {}
    for obj, brs in branch_dict.items():
        #print(obj, brs)
        if obj == 'L1Mu':
            #print(obj)
            # Rename 'hwEtaAtVtx' to 'hwEta' for L1Mu
            new_arr[obj] = ak.zip({
                'pt': events_raw[obj + "_hwPt"],
                'eta': events_raw[obj + "_hwEtaAtVtx"],  # Renamed here
                'phi': events_raw[obj + "_hwPhiAtVtx"]
            })
        elif obj == 'L1EtSum':
            #print(obj)
            # Rename 'hwEtaAtVtx' to 'hwEta' for L1Mu
            new_arr[obj] = ak.zip({
                'pt': events_raw[obj + "_hwPt"],
                'phi': events_raw[obj + "_hwPhi"],  # Renamed here
            })
        else: 
            new_arr[obj] = ak.zip({
                'pt': events_raw[obj + "_hwPt"],
                'eta': events_raw[obj + "_hwEta"],  # Renamed here
                'phi': events_raw[obj + "_hwPhi"]
        })
    events_hw = ak.Array(new_arr)
    
    n_events = len(events_hw)
    events_hw_flat = np.zeros((n_events, 57), dtype='int')
    
    events_hw_flat = np.zeros((len(events_hw), 57), dtype='int')
    events_hw_flat[:,0] = ak.to_numpy(ak.fill_none(ak.pad_none(events_hw.L1EtSum["pt"], 8, clip=True)[:,2], -1)).reshape(-1)  #  ak.to_numpy(ak.fill_none(ak.pad_none(events_hw.L1EtSum["pt"], 1, clip=True), -1)).reshape(-1)
    events_hw_flat[:,2] =  ak.to_numpy(ak.fill_none(ak.pad_none(events_hw.L1EtSum["phi"], 8, clip=True)[:,2], -1)).reshape(-1)   #ak.to_numpy(ak.fill_none(ak.pad_none(events_hw.L1EtSum["phi"], 1, clip=True), -1)).reshape(-1)
    events_hw_flat[:,3:3+3*(NEG)] = awkward_to_numpy(events_hw.L1EG, NEG)
    events_hw_flat[:,3+3*(NEG):3+3*(NMU+NEG)] = awkward_to_numpy(events_hw.L1Mu, NMU)
    events_hw_flat[:,3+3*(NMU+NEG):3+3*(NMU+NEG+NJET)] = awkward_to_numpy(events_hw.L1Jet, NJET)

    file_name = str(time.time_ns()) 
    file_name = file_name + "_" + str(int(random.random()*10000))
    file_name = file_name + "_" + str(int(random.random()*10000))
    
    file_name = os.path.join(save_dir,file_name)
    
    file = zarr.open_group(file_name+".zarr",
                           mode="w",
                           zarr_version=2)

    file["Data"] = events_hw_flat
    file["RunNumber"] = RunNumber
    file["LuminosityBlock"] = LuminosityBlock
    file["BunchCrossing"] = BunchCrossing
    file["OrbitNumber"] = OrbitNumber
    file["nl1EG"] = nL1EG
    file["nL1Jet"] = nL1Jet

    file.attrs["WriteTime"] = time.time_ns() ## Acts like a lock !
    file.store.close()

    return 0


    
