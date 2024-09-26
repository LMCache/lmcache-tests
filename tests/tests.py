import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from driver import run_test_case, run_test_cases
from test_cases import TestCase
from configs import BootstrapConfig, WorkloadConfig, Usecase
from configs import VLLMConfig, VLLMOptionalConfig, LMCacheConfig, EngineType


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



##### Test cases #####
def test_lmcache_local_gpu() -> pd.DataFrame:
    # Start two servers: with lmcache and without lmcache
    config1 = CreateSingleLocalBootstrapConfig(8000, 0, "mistralai/Mistral-7B-Instruct-v0.2", "configs/lmcache_local_gpu.yaml")
    config2 = CreateSingleLocalBootstrapConfig(8001, 1, "mistralai/Mistral-7B-Instruct-v0.2", None)

    # Experiments: 8K, 16K, 24K shared context, each experiments has 5 queries
    lengths = [8192, 16384, 24576]
    experiments = [CreateDummyExperiment(5, length) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config1, config2])

    # Run test case
    final_result = run_test_case(test_case)
    return final_result

def test_lmcache_local_cpu() -> pd.DataFrame:
    # Start two servers: with lmcache and without lmcache
    config1 = CreateSingleLocalBootstrapConfig(8000, 0, "mistralai/Mistral-7B-Instruct-v0.2", "configs/lmcache_local_cpu.yaml")
    config2 = CreateSingleLocalBootstrapConfig(8001, 1, "mistralai/Mistral-7B-Instruct-v0.2", None)

    # Experiments: 8K, 16K, 24K shared context, each experiments has 5 queries
    lengths = [8192, 16384, 24576]
    # lengths = [8192]
    experiments = [CreateDummyExperiment(10, length ) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config1, config2])

    # Run test case
    final_result = run_test_case(test_case)
    return final_result

def test_lmcache_local_disk() -> pd.DataFrame:
    # Start two servers: with lmcache and without lmcache
    config1 = CreateSingleLocalBootstrapConfig(8000, 0, "mistralai/Mistral-7B-Instruct-v0.2", "configs/lmcache_local_disk.yaml")
    config2 = CreateSingleLocalBootstrapConfig(8001, 1, "mistralai/Mistral-7B-Instruct-v0.2", None)

    # Experiments: 8K, 16K, 24K shared context, each experiments has 5 queries
    lengths = [8192, 16384, 24576]
    experiments = [CreateDummyExperiment(10, length) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config1, config2])

    # Run test case
    final_result = run_test_case(test_case)
    return final_result

def test_lmcache_local_distributed() -> pd.DataFrame:
    """
    Local CPU + TP = 2
    """
    config = CreateSingleLocalBootstrapConfig(8000, 0, "mistralai/Mistral-7B-Instruct-v0.2", "configs/lmcache_local_cpu.yaml")

    config.vllm_config.tensor_parallel_size = 2
    config.vllm_config.gpu_memory_utilization = 0.6
    config.envs = {}

    # Experiments: 8K, 16K, 24K shared context, each experiments has 5 queries
    lengths = [8192, 16384, 24576]
    experiments = [CreateDummyExperiment(10, length) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config])

    # Run test case
    final_result = run_test_case(test_case)
    return final_result

def test_lmcache_remote_cachegen() -> pd.DataFrame:
    # Start two servers: with lmcache and without lmcache
    config1 = CreateSingleLocalBootstrapConfig(8000, 0, "mistralai/Mistral-7B-Instruct-v0.2", "configs/lmcache_remote_cachegen.yaml")
    config2 = CreateSingleLocalBootstrapConfig(8001, 1, "mistralai/Mistral-7B-Instruct-v0.2", "configs/lmcache_remote_cachegen_pipeline.yaml")

    # Experiments: 8K, 16K, 24K shared context, each experiments has 5 queries
    lengths = [8192, 16384, 24576]
    experiments = [CreateDummyExperiment(10, length) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config1, config2])

    # Run test case
    final_result = run_test_case(test_case)
    return final_result

def test_lmcache_cachegen_distributed() -> pd.DataFrame:
    config = CreateSingleLocalBootstrapConfig(8000, 0, "mistralai/Mistral-7B-Instruct-v0.2", "configs/lmcache_remote_cachegen.yaml")
    #config = CreateSingleLocalBootstrapConfig(8000, 0, "facebook/opt-125m", "configs/lmcache_remote_cachegen.yaml")

    config.vllm_config.tensor_parallel_size = 2
    config.vllm_config.gpu_memory_utilization = 0.6
    config.envs = {"CUDA_LAUNCH_BLOCKING": "0"}

    # Experiments: 8K, 16K, 24K shared context, each experiments has 5 queries
    lengths = [16384, 24576]
    experiments = [CreateDummyExperiment(10, length, gap_between_requests=8) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config])

    # Run test case
    final_result = run_test_case(test_case)
    return final_result

def test_lmcache_remote_safetensor() -> pd.DataFrame:
    # Start two servers: with lmcache and without lmcache
    config1 = CreateSingleLocalBootstrapConfig(8000, 0, "mistralai/Mistral-7B-Instruct-v0.2", "configs/lmcache_remote_safetensor.yaml")
    config2 = CreateSingleLocalBootstrapConfig(8001, 1, "mistralai/Mistral-7B-Instruct-v0.2", "configs/lmcache_remote_safetensor_pipeline.yaml")

    # Experiments: 8K, 16K, 24K shared context, each experiments has 5 queries
    lengths = [8192, 16384, 24576]
    experiments = [CreateDummyExperiment(10, length) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config1, config2])

    # Run test case
    final_result = run_test_case(test_case)
    return final_result

def test_lmcache_safetensor_distributed() -> pd.DataFrame:
    """
    LMCache remote safetensor + TP = 2
    """
    config = CreateSingleLocalBootstrapConfig(8000, 0, "mistralai/Mistral-7B-Instruct-v0.2", "configs/lmcache_remote_safetensor.yaml")

    config.vllm_config.tensor_parallel_size = 2
    config.vllm_config.gpu_memory_utilization = 0.6
    config.envs = {}

    # Experiments: 8K, 16K, 24K shared context, each experiments has 5 queries
    lengths = [8192, 16384, 24576]
    experiments = [CreateDummyExperiment(10, length) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config])

    # Run test case
    final_result = run_test_case(test_case)
    return final_result

def test_lmcache_remote_disk() -> pd.DataFrame:
    # Start two servers: with lmcache and without lmcache
    config1 = CreateSingleLocalBootstrapConfig(8000, 0, "mistralai/Mistral-7B-Instruct-v0.2", "configs/lmcache_remote_cachegen.yaml")
    config2 = CreateSingleLocalBootstrapConfig(8002, 1, "mistralai/Mistral-7B-Instruct-v0.2", None)

    config1.lmcache_config.remote_device = "/local/lmcache-tests/lmcache-server"

    # Experiments: 8K, 16K, 24K shared context, each experiments has 5 queries
    lengths = [8192, 16384, 24576]
    experiments = [CreateDummyExperiment(10, length) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config1, config2])

    # Run test case
    final_result = run_test_case(test_case)
    return final_result

def test_lmcache_chatglm() -> pd.DataFrame:
    # Start two servers: with lmcache and without lmcache
    config1 = CreateSingleLocalBootstrapConfig(8000, 0, "THUDM/glm-4-9b-chat", "configs/lmcache_remote_cachegen.yaml")
    config2 = CreateSingleLocalBootstrapConfig(8001, 1, "THUDM/glm-4-9b-chat", None)

    config1.vllm_config.tensor_parallel_size = 2
    config1.vllm_config.gpu_memory_utilization = 0.8
    config1.envs = {}
    config1.vllm_optional_config["trust_remote_code"] = ""

    config2.vllm_config.tensor_parallel_size = 2
    config2.vllm_config.gpu_memory_utilization = 0.8
    config2.envs = {}
    config2.vllm_optional_config["trust_remote_code"] = ""

    # Experiments: 8K, 16K, 24K shared context, each experiments has 5 queries
    lengths = [8192, 16384, 24576]
    experiments = [CreateDummyExperiment(10, length) for length in lengths]

    test_case1 = TestCase(
            experiments = experiments,
            engines = [config1])

    test_case2 = TestCase(
            experiments = experiments,
            engines = [config2])

    # Run test case
    return run_test_cases([test_case1, test_case2])

def test_lmcache_redis_sentinel() -> pd.DataFrame:
    # Start two servers: with lmcache and without lmcache
    config1 = CreateSingleLocalBootstrapConfig(8000, 1, "mistralai/Mistral-7B-Instruct-v0.2", "configs/lmcache_redis_sentinel_cachegen.yaml")

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
