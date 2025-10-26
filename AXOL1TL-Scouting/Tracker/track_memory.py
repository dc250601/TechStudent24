import os
import subprocess
import time

def get_folder_size(path):
    if os.path.exists(path):
        result = subprocess.run(['du', '-s', '--block-size=1G', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Error: {result.stderr.strip()}")
        size_str = result.stdout.split()[0]
    else:
        size_str = "0"
    return size_str


if __name__ == "__main__":
    while(True):
        try:
            stage0_temp_mem = get_folder_size("/dev/shm/temp_stage0/")
            stage0_mem = get_folder_size("/dev/shm/StoreStage0/")
            stage1_mem = get_folder_size("/dev/shm/StoreStage1/")
            stage2_mem = get_folder_size("/dev/shm/StoreStage2/")
            stage3_mem = get_folder_size("/dev/shm/StoreStage3/")
            now = time.gmtime()
            formatted_time = time.strftime(f"%Y-%m-%d %H:%M:%S UTC|Stage-0 (temp) Mem {stage0_temp_mem}GB|Stage-0 Mem {stage0_mem}GB|Stage-1 Mem {stage1_mem}GB|Stage-2 Mem {stage2_mem}GB|Stage-3 Mem {stage3_mem}GB|", now)
            print(formatted_time)
            time.sleep(5)
        except:
            now = time.gmtime()
            formatted_time = time.strftime(f"%Y-%m-%d %H:%M:%S UTC|Error in memory tracking|", now)
            print(formatted_time)
            time.sleep(5)
            continue

    