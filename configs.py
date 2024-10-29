from typing import Optional, Dict, Union, List
from enum import Enum
from dataclasses import dataclass, asdict
import json

@dataclass
class Config:
    def __getitem__(self, key: str) -> int:
        return getattr(self, key)

class Usecase(Enum):
    """
    Specification for the use case
    - DUMMY: the dummy use case, for basic testing
    - PREFIX_REUSE: the prefix reuse use case
    - RAG: the RAG use case
    - MULTI: the multi-turn conversation use case
    - VARY: the variable length use case
    """
    DUMMY = 1
    PREFIX_REUSE = 2
    RAG = 3
    MULTI = 4
    VARY = 5

@dataclass
class WorkloadConfig(Config):
    """
    - QPS: query per second
    - Duration: number of seconds
    - Context length: number of tokens in the requests (approximate number)
    - Query length
    """
    # Number of queries per second
    qps: int

    # Total duration of the workload in seconds 
    duration: float

    # Number of tokens in the context (approximate number)
    context_length: int

    # Number of tokens in the suffix question
    query_length: int

    # Offset of the timestamps
    offset: float
    
    def desc(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class LMCacheConfig(Config):
    # Path to the lmcache configuration
    config_path: str

    remote_device: Optional[str] = None

    def cmdargs(self) -> str:
        return " " if self.config_path is not None else ""
        # return f"--lmcache-config-file {self.config_path}" if self.config_path is not None else ""

@dataclass
class VLLMConfig(Config):
    # which Model is used
    model: str

    # vLLM engine's port 
    port: int

    # Memory limit for the vLLM engine
    gpu_memory_utilization: float

    # Tensor parallelism
    tensor_parallel_size: Optional[int]

    def cmdargs(self) -> str:
        args = []
        for key, value in self.__dict__.items():
            if value is None:
                continue
            if key=="model":
                args.append(f"{value}")
                continue
            modified_key = key.replace("_", "-")
            args.append(f"--{modified_key} {value}")

        return " ".join(args)

@dataclass
class VLLMOptionalConfig(Config):
    """
    Optional cmdline configuration for the vLLM engine
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setitem__(self, key: str, value: int):
        setattr(self, key, value)

    def __str__(self):
        return "VLLMOptionalConfig({})".format(
                ", ".join([f"{k}={v}" for k, v in self.__dict__.items()]))

    def __repr__(self):
        return self.__str__()

    def cmdargs(self) -> str:
        args = []
        for key, value in self.__dict__.items():
            if value is None:
                continue
            modified_key = key.replace("_", "-")
            args.append(f"--{modified_key} {value}")

        return " ".join(args)

class EngineType(Enum):
    """
    What kind of engine will be bootstraped
    """
    LOCAL = 1
    DOCKER = 2

@dataclass
class BootstrapConfig:
    # What kind of engine will be bootstraped
    # TODO: this should be in the test case specification
    engine_type: EngineType

    # Required VLLM configurations
    vllm_config: VLLMConfig

    # Optional VLLM configurations
    vllm_optional_config: VLLMOptionalConfig

    # LMCache configurations
    lmcache_config: LMCacheConfig

    # Extra environment variables
    envs: Dict[str, str]


# TODO: configuration loader




#if __name__ == "__main__":
#    # Example usage
#    workload_config = WorkloadConfig(qps=100, duration=10, context_length=100, query_length=10)
#    lmcache_config = LMCacheConfig(config_path="configs/lmcache_config.yaml")
#    vllm_config = VLLMConfig(port=8000, model="gpt2", gpu_memory_utilization=0.5, tensor_parallelism=1)
#    vllm_optional_config = VLLMOptionalConfig(**{"key1": 1, "key2": 2})
#    vllm_optional_config["key3"] = 3
#
#    print(workload_config)
#    print(lmcache_config)
#    print(vllm_config)
#    print(vllm_optional_config)
#    print(vllm_config.cmdargs())
#    print(vllm_optional_config.cmdargs())
