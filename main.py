import pandas as pd

from driver import run_test_case
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
            gpu_memory_utilization = 0.5,
            tensor_parallel_size = 1),
        vllm_optional_config = VLLMOptionalConfig(),
        lmcache_config = LMCacheConfig(lmcache_config_path),
        envs = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    )

def CreateDummyExperiment(num_requests, context_length, gap_between_requests = 10):
    """
    Create some requests for DUMMY usecase
    The query length will be 16
    """
    qps = 1 / gap_between_requests
    duration = num_requests * gap_between_requests
    cfg = WorkloadConfig(qps, duration, context_length, 16, offset = 0)
    return (cfg, Usecase.DUMMY)

def wrapped_runner(test_func, output_file):
    """
    Run the test case and save results to output file
    """
    output_df = test_func()
    output_df.to_csv(output_file, index=False)

def process_result(result_csv: str):
    """
    Generate folloing plots:
    - LINE, y: TTFT, x: request ID, color by engine id, graph by expr_id
    - LINE, y: throughput, x: request ID, color by engine id, graph by expr_id
    - BAR, y: GPU memory, x: expr_id, color by engine id
    """
    df = pd.read_csv(result_csv)

    from matplotlib import pyplot as plot
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_pages = PdfPages(result_csv.replace(".csv", ".pdf"))
    df.groupby(["expr_id", "engine_id"]).plot(x="request_id", y="TTFT", title="TTFT", ax=plot.gca())

    pdf_pages.close()


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
    experiments = [CreateDummyExperiment(5, length) for length in lengths]

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
    experiments = [CreateDummyExperiment(5, length) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config1, config2])

    # Run test case
    final_result = run_test_case(test_case)
    return final_result

def test_lmcache_remote_cachegen() -> pd.DataFrame:
    # Start two servers: with lmcache and without lmcache
    config1 = CreateSingleLocalBootstrapConfig(8000, 0, "mistralai/Mistral-7B-Instruct-v0.2", "configs/lmcache_remote_cachegen.yaml")
    config2 = CreateSingleLocalBootstrapConfig(8001, 1, "mistralai/Mistral-7B-Instruct-v0.2", "configs/lmcache_remote_cachegen_pipeline.yaml")

    # Experiments: 8K, 16K, 24K shared context, each experiments has 5 queries
    lengths = [8192, 16384, 24576]
    experiments = [CreateDummyExperiment(5, length) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config1, config2])

    # Run test case
    final_result = run_test_case(test_case)
    return final_result

def test_lmcache_remote_safetensor() -> pd.DataFrame:
    # Start two servers: with lmcache and without lmcache
    config1 = CreateSingleLocalBootstrapConfig(8000, 0, "mistralai/Mistral-7B-Instruct-v0.2", "configs/lmcache_remote_safetensor.yaml")
    config2 = CreateSingleLocalBootstrapConfig(8001, 1, "mistralai/Mistral-7B-Instruct-v0.2", "configs/lmcache_remote_safetensor_pipeline.yaml")

    # Experiments: 8K, 16K, 24K shared context, each experiments has 5 queries
    lengths = [8192, 16384, 24576]
    experiments = [CreateDummyExperiment(5, length) for length in lengths]

    test_case = TestCase(
            experiments = experiments,
            engines = [config1, config2])

    # Run test case
    final_result = run_test_case(test_case)
    return final_result


if __name__ == "__main__":
    print("Start running test cases")
    #wrapped_runner(test_lmcache_local_gpu, "outputs/test_lmcache_local_gpu.csv")
    #wrapped_runner(test_lmcache_local_cpu, "outputs/test_lmcache_local_cpu.csv")
    wrapped_runner(test_lmcache_local_disk, "outputs/test_lmcache_local_disk.csv")
    wrapped_runner(test_lmcache_remote_cachegen, "outputs/test_lmcache_remote_cachegen.csv")
    wrapped_runner(test_lmcache_remote_cachegen, "outputs/test_lmcache_remote_safetensor.csv")
