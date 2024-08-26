from dataclasses import dataclass
from typing import List, Tuple

from configs import WorkloadConfig, BootstrapConfig, Usecase

@dataclass
class TestCase:
    """
    Each test case has a list of N workloads and a list of M engines
    It will run N experiments for each workload.
    Wihtin a single experiment, it will run M engines, and create M workload generators for each engine.
    """
    experiments: List[Tuple[WorkloadConfig, Usecase]]
    engines: List[BootstrapConfig]

def CreateTestCase():
    """
    Helper function to create test case easily
    """
    raise NotImplementedError

