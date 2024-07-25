from multiprocessing import Process
import argparse
import subprocess
import time
from utils.setup import start_vllm, start_shared_storage

parser = argparse.ArgumentParser(description='setup')

parser.add_argument(
    '--node', 
    choices=['remote_shared', 'vllm_optimized', 'vllm_original', 'all', 'all_local'])
parser.add_argument(
    '--cuda_debug', 
    action='store_true', 
    default=False)

args = parser.parse_args()
node = args.node
cuda_debug = args.cuda_debug

if node == "vllm_original":
    start_vllm(
        port=8000,
        cuda_devices=[0], 
        enable_remote=False, 
        enable_cache=False, 
        sync=True,
        cuda_debug=cuda_debug)
elif node == "vllm_optimized":
    start_vllm(
        port=8001,
        cuda_devices=[1], 
        enable_remote=True, 
        enable_cache=True, 
        sync=True,
        cuda_debug=cuda_debug)
elif node == "remote_shared":
    start_shared_storage(65432, sync=True)
elif node in ['all', 'all_local']:
    proc_storage = start_shared_storage(65432, sync=False)
    
    proc_vllm = start_vllm(
        port=8000, 
        cuda_devices=[0], 
        enable_remote=False, 
        enable_cache=False, 
        sync=False,
        cuda_debug=cuda_debug)
    
    remote = True if node=="all" else False
    proc_vllm_optimized = start_vllm(
        port=8001, 
        cuda_devices=[1], 
        enable_remote=remote, 
        enable_cache=True, 
        sync=False,
        cuda_debug=cuda_debug)
    processes = [proc_storage, proc_vllm, proc_vllm_optimized]
    '''
    try:
        while True:
            time.sleep(1000)
    except KeyboardInterrupt:
        print("killing processes")
        for process in processes:
            process.kill()
    '''