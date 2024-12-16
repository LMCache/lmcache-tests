import pandas as pd
import sys, os
import time, re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from driver import run_test_case, run_test_cases
from test_cases import TestCase
from configs import BootstrapConfig, WorkloadConfig, Usecase
from configs import VLLMConfig, VLLMOptionalConfig, LMCacheConfig, EngineType
from utils import run_command, get_max_context_length
import yaml

##### Helper functions #####
def CreateSingleLocalBootstrapConfig(
        port: int,
        gpu_id: int,
        model: str,
        lmcache_config_path: str
    ) -> BootstrapConfig:

    return BootstrapConfig(
        engine_type=EngineType.LOCAL,
        vllm_config = VLLMConfig(
            port = port,
            model = model,
            gpu_memory_utilization = 0.8,
            tensor_parallel_size = 1),
        vllm_optional_config = VLLMOptionalConfig(),
        lmcache_config = LMCacheConfig(lmcache_config_path),
        envs = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    )

def CreateDummyExperiment(num_requests, context_length, gap_between_requests = 8):
    """
    Create some requests for DUMMY usecase
    The query length will be 16
    """
    qps = 1 / gap_between_requests
    duration = num_requests * gap_between_requests
    cfg = WorkloadConfig(qps, duration, context_length, 16, offset = 0)
    return (cfg, Usecase.DUMMY)

def CreateMultiTurnExperiment(num_requests, context_length, gap_between_requests = 8):
    """
    Create some requests for multi-turn conversation
    The query length will be 16
    """
    qps = 1 / gap_between_requests
    duration = num_requests * gap_between_requests
    cfg = WorkloadConfig(qps, duration, context_length, 16, offset = 0)
    return (cfg, Usecase.MULTI)

def CreateVaryLengthExperiment(num_requests, context_length, gap_between_requests = 8):
    """
    Create some requests for variable length usecase
    The query length will be 16
    """
    qps = 1 / gap_between_requests
    duration = num_requests * gap_between_requests
    cfg = WorkloadConfig(qps, duration, context_length, 16, offset = 0)
    return (cfg, Usecase.VARY)

def ModelConfig(model: str, BootstrapConfig) -> None:
    """
    Set configuration for bootstrap for different models
    """
    match model:
        case "mistralai/Mistral-7B-Instruct-v0.2":
            pass
        case "THUDM/glm-4-9b-chat":
            BootstrapConfig.vllm_config.tensor_parallel_size = 2
            BootstrapConfig.vllm_optional_config["trust_remote_code"] = ""
            BootstrapConfig.vllm_optional_config["enable_chunked_prefill"] = False
            BootstrapConfig.vllm_optional_config["max_model_len"] = get_max_context_length(model)
        case "meta-llama/Llama-3.1-8B-Instruct":
            BootstrapConfig.vllm_optional_config["enable_chunked_prefill"] = False
            BootstrapConfig.vllm_optional_config["max_model_len"] = get_max_context_length(model)
        case _:
            pass


##### Test cases #####
def offline_test(model = "mistralai/Mistral-7B-Instruct-v0.2") -> pd.DataFrame:
    """
    This function tests partial prefll and full prefill in a single batch.
    """
    user_name=os.popen('whoami').read()[:-1]
    stdout_log = os.path.join(f"/tmp/{user_name}-65431-stdout.log")
    stderr_log = os.path.join(f"/tmp/{user_name}-65431-stderr.log")
    run_command("lmcache_server localhost 65431", stdout_log, None, detach=True)
    time.sleep(10)
    os.environ['LMCACHE_CONFIG_FILE'] = "./configs/example.yaml"
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    run_command("python3 tests/offline_test.py", None, stderr_log, detach=False)
    return None

def test_cache_compatibility(model = "mistralai/Mistral-7B-Instruct-v0.2") -> pd.DataFrame:
    """
    This function tests the compatibility of prefix caching between lmcache and vllm 
    by comparing performance across enabling prefix caching or not.
    """
    # Start two servers with lmcache enabling prefix caching
    config1 = CreateSingleLocalBootstrapConfig(8000, 0, model, "configs/lmcache_local_cpu.yaml")
    config2 = CreateSingleLocalBootstrapConfig(8001, 1, model, "configs/lmcache_local_cpu.yaml")
    config1.vllm_optional_config["enable_prefix_caching"] = ""

    # Set vllm configuration for different models
    ModelConfig(model, config1)
    ModelConfig(model, config2)

    # Experiments: 8K, 16K, 24K shared context, each experiments has 5 queries
    lengths = [8192, 16384, 24576]
    experiments = [CreateDummyExperiment(5, length ) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config1, config2])

    # Run test case
    final_result = run_test_case(test_case)
    return final_result

def test_chunk_prefill(model = "mistralai/Mistral-7B-Instruct-v0.2") -> pd.DataFrame:
    """
    This function tests the performance of chunked prefill by comparing scenarios with and without lmcache.
    """
    # Start two servers: with lmcache and without lmcache
    config1 = CreateSingleLocalBootstrapConfig(8000, 0, model, "configs/lmcache_local_cpu.yaml")
    config2 = CreateSingleLocalBootstrapConfig(8001, 1, model, None)
    config1.vllm_optional_config["enable_chunked_prefill"] = True
    config2.vllm_optional_config["enable_chunked_prefill"] = True

    # Set vllm configuration for different models
    ModelConfig(model, config1)
    ModelConfig(model, config2)

    # Experiments: 8K, 16K, 24K shared context, each experiments has 5 queries
    lengths = [8192, 16384, 24576]
    experiments = [CreateDummyExperiment(5, length ) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config1, config2])

    # Run test case
    final_result = run_test_case(test_case)
    return final_result

def test_vary_length_workload(model = "mistralai/Mistral-7B-Instruct-v0.2") -> pd.DataFrame:
    """
    This function tests the performance of partial prefill by changing the workload length of different requests.
    """
    # Start one server with lmcache
    config = CreateSingleLocalBootstrapConfig(8000, 0, model, "configs/lmcache_local_cpu.yaml")

    # Set vllm configuration for different models
    ModelConfig(model, config)

    # One experiments: context length sequence 12K, 24K, 24K
    lengths = [12288]
    experiments = [CreateVaryLengthExperiment(3, length ) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config])

    # Run test case
    final_result = run_test_case(test_case)
    return final_result

def test_multi_turn(model = "mistralai/Mistral-7B-Instruct-v0.2") -> pd.DataFrame:
    """
    This function tests the performance of saving decode KV Cache with a multi-turn conversation
    by comparing performance with and without lmcache.
    """
    # Start one server: with lmcache; for contrast (not saving decode KV Cache), change save_decode_cache to false
    config1 = CreateSingleLocalBootstrapConfig(8000, 0, model, "configs/lmcache_local_cpu.yaml")
    config2 = CreateSingleLocalBootstrapConfig(8001, 1, model, "configs/lmcache_local_cpu_multi.yaml")

    # Set vllm configuration for different models
    ModelConfig(model, config1)
    ModelConfig(model, config2)

    # Experiment: ONE query that contains 10 rounds 
    lengths = [16] # useless for this test case
    experiments = [CreateMultiTurnExperiment(1, length ) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config1, config2])
    
    # Run test case
    final_result = run_test_case(test_case)
    return final_result

def test_lmcache_local_gpu(model = "mistralai/Mistral-7B-Instruct-v0.2") -> pd.DataFrame:
    """
    This function tests local gpu storage backend by comparing scenarios with and without lmcache.
    """
    # Start two servers: with lmcache and without lmcache
    config1 = CreateSingleLocalBootstrapConfig(8000, 0, model, "configs/lmcache_local_gpu.yaml")
    config2 = CreateSingleLocalBootstrapConfig(8001, 1, model, None)

    # Set vllm configuration for different models
    ModelConfig(model, config1)
    ModelConfig(model, config2)

    # Experiments: 8K, 16K, 24K shared context, each experiments has 10 queries
    lengths = [8192, 16384, 24576]
    experiments = [CreateDummyExperiment(5, length) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config1, config2])

    # Run test case
    final_result = run_test_case(test_case)
    return final_result

def test_lmcache_local_cpu(model = "mistralai/Mistral-7B-Instruct-v0.2") -> pd.DataFrame:
    """
    This function tests local cpu storage backend by comparing scenarios with and without lmcache.
    """
    # Start two servers: with lmcache and without lmcache
    config1 = CreateSingleLocalBootstrapConfig(8000, 0, model, "configs/lmcache_local_cpu.yaml")
    config2 = CreateSingleLocalBootstrapConfig(8001, 1, model, None)

    # Set vllm configuration for different models
    ModelConfig(model, config1)
    ModelConfig(model, config2)

    # Experiments: 8K, 16K, 24K shared context, each experiments has 10 queries
    lengths = [8192, 16384, 24576]
    experiments = [CreateDummyExperiment(10, length ) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config1, config2])

    # Run test case
    final_result = run_test_case(test_case)
    return final_result

def test_lmcache_local_disk(model = "mistralai/Mistral-7B-Instruct-v0.2") -> pd.DataFrame: 
    """
    This function tests local disk storage backend by comparing scenarios with and without lmcache.
    """
    # Start two servers: with lmcache and without lmcache
    yaml_config = "configs/lmcache_local_disk.yaml"
    config1 = CreateSingleLocalBootstrapConfig(8000, 0, model, yaml_config)
    config2 = CreateSingleLocalBootstrapConfig(8001, 1, model, None)

    # Set vllm configuration for different models
    ModelConfig(model, config1)
    ModelConfig(model, config2)

    # Experiments: 8K, 16K, 24K shared context, each experiments has 5 queries
    # lengths = [8192, 16384, 24576]
    lengths = [8192]
    experiments = [CreateDummyExperiment(1, length) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config1, config2])

    # Run test case
    final_result = run_test_case(test_case)

    # Clean up
    with open(yaml_config, 'r') as file:
        data = yaml.safe_load(file)
    local_device = data.get('local_device') + "*"
    os.system(f"rm -rf {local_device}")

    return final_result

def test_lmcache_local_distributed(model = "mistralai/Mistral-7B-Instruct-v0.2") -> pd.DataFrame: 
    """
    Local CPU + TP = 2
    """
    config = CreateSingleLocalBootstrapConfig(8000, 0, model, "configs/lmcache_local_cpu.yaml")

    config.vllm_config.tensor_parallel_size = 2
    config.vllm_config.gpu_memory_utilization = 0.6
    config.envs = {}

    # Set vllm configuration for different models
    ModelConfig(model, config)

    # Experiments: 8K, 16K, 24K shared context, each experiments has 5 queries
    lengths = [8192, 16384, 24576]
    experiments = [CreateDummyExperiment(10, length) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config])

    # Run test case
    final_result = run_test_case(test_case)
    return final_result

def test_lmcache_remote_cachegen(model = "mistralai/Mistral-7B-Instruct-v0.2") -> pd.DataFrame:
    """
    This function is set with local cpu storage backend and cachegen for transmission 
    by comparing scenarios whether retrieval is pipelined or not.
    """
    # Start two servers
    config1 = CreateSingleLocalBootstrapConfig(8000, 0, model, "configs/lmcache_remote_cachegen.yaml")
    config2 = CreateSingleLocalBootstrapConfig(8001, 1, model, "configs/lmcache_remote_cachegen_pipeline.yaml")

    # Set vllm configuration for different models
    ModelConfig(model, config1)
    ModelConfig(model, config2)
    
    # Experiments: 8K, 16K, 24K shared context, each experiments has 5 queries
    lengths = [8192, 16384, 24576]
    experiments = [CreateDummyExperiment(10, length) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config1, config2])

    # Run test case
    final_result = run_test_case(test_case)
    return final_result

def test_lmcache_cachegen_distributed(model = "mistralai/Mistral-7B-Instruct-v0.2") -> pd.DataFrame:
    """
    This function is set with local cpu storage backend and cachegen for transmission 
    by enabling distributed cuda.
    """
    config = CreateSingleLocalBootstrapConfig(8000, 0, model, "configs/lmcache_remote_cachegen.yaml")
    #config = CreateSingleLocalBootstrapConfig(8000, 0, "facebook/opt-125m", "configs/lmcache_remote_cachegen.yaml")

    config.vllm_config.tensor_parallel_size = 2
    config.vllm_config.gpu_memory_utilization = 0.6
    config.envs = {"CUDA_LAUNCH_BLOCKING": "0"}

    # Set vllm configuration for different models
    ModelConfig(model, config)

    # Experiments: 16K, 24K shared context, each experiments has 10 queries
    lengths = [16384, 24576]
    experiments = [CreateDummyExperiment(10, length, gap_between_requests=8) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config])

    # Run test case
    final_result = run_test_case(test_case)
    return final_result

def test_lmcache_remote_safetensor(model = "mistralai/Mistral-7B-Instruct-v0.2") -> pd.DataFrame:
    """
    This function is set with local cpu storage backend and safetensor for transmission 
    by comparing scenarios whether retrieval is pipelined or not.
    """
    # Start two servers: with lmcache and without lmcache
    config1 = CreateSingleLocalBootstrapConfig(8000, 0, model, "configs/lmcache_remote_safetensor.yaml")
    config2 = CreateSingleLocalBootstrapConfig(8001, 1, model, "configs/lmcache_remote_safetensor_pipeline.yaml")

    # Set vllm configuration for different models
    ModelConfig(model, config1)
    ModelConfig(model, config2)
    
    # Experiments: 8K, 16K, 24K shared context, each experiments has 5 queries
    lengths = [8192, 16384, 24576]
    experiments = [CreateDummyExperiment(10, length) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config1, config2])

    # Run test case
    final_result = run_test_case(test_case)
    return final_result

def test_lmcache_safetensor_distributed(model = "mistralai/Mistral-7B-Instruct-v0.2") -> pd.DataFrame:
    """
    LMCache remote safetensor + TP = 2
    """
    config = CreateSingleLocalBootstrapConfig(8000, 0, model, "configs/lmcache_remote_safetensor.yaml")

    config.vllm_config.tensor_parallel_size = 2
    config.vllm_config.gpu_memory_utilization = 0.6
    config.envs = {}

    # Set vllm configuration for different models
    ModelConfig(model, config)

    # Experiments: 8K, 16K, 24K shared context, each experiments has 5 queries
    lengths = [8192, 16384, 24576]
    experiments = [CreateDummyExperiment(10, length) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config])

    # Run test case
    final_result = run_test_case(test_case)
    return final_result

def test_lmcache_remote_disk(model = "mistralai/Mistral-7B-Instruct-v0.2") -> pd.DataFrame:
    """
    This function tests remote disk storage backend by comparing scenarios with and without lmcache.
    """
    # Start two servers: with lmcache and without lmcache
    config1 = CreateSingleLocalBootstrapConfig(8000, 0, model, "configs/lmcache_remote_cachegen.yaml")
    config2 = CreateSingleLocalBootstrapConfig(8001, 1, model, None)

    config1.lmcache_config.remote_device = "/local/end-to-end-tests/lmcache-server/"

    # Set vllm configuration for different models
    ModelConfig(model, config1)
    ModelConfig(model, config2)

    # Experiments: 8K, 16K, 24K shared context, each experiments has 5 queries
    lengths = [8192, 16384, 24576]
    experiments = [CreateDummyExperiment(10, length) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config1, config2])

    # Run test case
    final_result = run_test_case(test_case)

    # Clean up
    os.system(f"rm -rf {config1.lmcache_config.remote_device}*")

    return final_result

def test_lmcache_redis_sentinel(model = "mistralai/Mistral-7B-Instruct-v0.2") -> pd.DataFrame:
    config1 = CreateSingleLocalBootstrapConfig(8000, 1, model, "configs/lmcache_redis_sentinel_cachegen.yaml")

    # Set vllm configuration for different models
    ModelConfig(model, config1)
    
    # Experiments: 8K, 16K, 24K shared context, each experiments has 5 queries
    #lengths = [8192, 16384, 24576]
    lengths = [24576]
    experiments = [CreateDummyExperiment(10, length) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config1])

    # Run test case
    final_result = run_test_case(test_case)
    return final_result

def dumb_test() -> None:
    print("This is a dumb_test") 

if __name__ == "__main__":
    print("Start running test cases")
    #wrapped_runner(test_lmcache_local_gpu, "outputs/test_lmcache_local_gpu.csv")
    #wrapped_runner(test_lmcache_local_cpu, "outputs/test_lmcache_local_cpu.csv")
    #wrapped_runner(test_lmcache_local_disk, "outputs/test_lmcache_local_disk.csv")
    #wrapped_runner(test_lmcache_local_distributed, "outputs/test_lmcache_local_distributed.csv")

    #wrapped_runner(test_lmcache_remote_safetensor, "outputs/test_lmcache_remote_safetensor.csv")
    #wrapped_runner(test_lmcache_safetensor_distributed, "outputs/test_lmcache_safetensor_distributed.csv")
    #wrapped_runner(test_lmcache_remote_cachegen, "outputs/test_lmcache_remote_cachegen.csv")
    #wrapped_runner(test_lmcache_cachegen_distributed, "outputs/test_lmcache_cachegen_distributed.csv")
    #wrapped_runner(test_lmcache_remote_disk, "outputs/test_lmcache_remote_disk.csv")

    #wrapped_runner(test_lmcache_chatglm, "outputs/test_lmcache_chatglm.csv")
