import os
import yaml
import subprocess

LMCACHE_LOCAL_CONFIG_PATH = "configs/lmcache_local.yaml"
LMCACHE_CONFIG_PATH = "configs/lmcache.yaml"
VLLM_CONFIG_PATH = "configs/vllm.yaml"

def on_exit(pid):
    def handler(signum, frame):
        os.kill(pid, signal.SIGTERM)
    return handler

def start_vllm(
    port, 
    cuda_devices=[0], 
    enable_remote=False, 
    enable_cache=False, 
    sync=True,
    cuda_debug=False):
    lmcache_config_path = LMCACHE_CONFIG_PATH if enable_remote else LMCACHE_LOCAL_CONFIG_PATH
    with open(VLLM_CONFIG_PATH, 'r') as fin:
        vllm_config = yaml.safe_load(fin)
    
    model_name = vllm_config.get("model_name", "mistralai/Mistral-7B-Instruct-v0.2")
    gpu_util = vllm_config.get("gpu_util", 0.5)
    tensor_parallelism = vllm_config.get("tensor_parallelism", 1)
    max_model_len = vllm_config.get("max_model_len", 32768)
    cmd = f"python3 -m vllm.entrypoints.openai.api_server \
        --model {model_name} \
        --port {port} \
        --gpu-memory-utilization {gpu_util} \
        --max-model-len {max_model_len} "
    if enable_cache:
        cmd += f"--lmcache-config-file {lmcache_config_path} "
    if tensor_parallelism > 1:
        cmd += f"--tensor-parallel-size {tensor_parallelism} "
    cmd += "--trust-remote-code"
    device_cmd = f"CUDA_VISIBLE_DEVICES="
    for device in cuda_devices:
        device_cmd += f"{device},"
    cmd = device_cmd + " " + cmd
    if cuda_debug:
        cmd = "CUDA_LAUNCH_BLOCKING=1 " + cmd
    print(f"Running command: {cmd}")
    if sync:
        os.system(cmd)
    else:
        proc = subprocess.Popen(cmd, shell=True, start_new_session=True)
        return proc

def start_shared_storage(port, sync=True):
    cmd = f"python3 -m lmcache_server.server localhost {port}"
    print(f"Running command: {cmd}")
    if sync:
        os.system(cmd)
    else:
        proc = subprocess.Popen(cmd, shell=True, start_new_session=True)
        return proc
    