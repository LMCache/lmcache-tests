import abc
import time
import subprocess
import os

from configs import LMCacheConfig, VLLMConfig, VLLMOptionalConfig, BootstrapConfig, EngineType
from utils import run_command
import log

logger = log.init_logger(__name__)

class Bootstrapper(metaclass=abc.ABCMeta):
    def __init__(self, config: BootstrapConfig):
        self.config = config

    @abc.abstractmethod
    def start(self):
        """
        Non-blocking function to start the serving engine
        """
        pass

    @abc.abstractmethod
    def wait_until_ready(self) -> bool:
        """
        Block until the serving engine is ready or it's dead
        Returns true if it's ready, otherwise false
        """

    @abc.abstractmethod
    def is_healthy(self) -> bool:
        """
        Non-block function to check if the engine is alive or not
        """

    @abc.abstractmethod
    def close(self):
        """
        Blocking function to shutdown the serving engine
        """
        pass

def CreateBootstrapper(config: BootstrapConfig, log_dir = "/tmp") -> Bootstrapper:
    match config.engine_type:
        case EngineType.LOCAL:
            return LocalVllmBootstrapper(config, log_dir)
        case _:
            raise ValueError(f"Unsupported engine type: {config.engine_type}")

class LocalVllmBootstrapper(Bootstrapper):
    def __init__(self, config: BootstrapConfig, log_dir = "/tmp"):
        super().__init__(config)
        self.command = self.get_cmdline()
        self.stdout_log = os.path.join(log_dir, f"{self.config.vllm_config.port}-stdout.log")
        self.stderr_log = os.path.join(log_dir, f"{self.config.vllm_config.port}-stderr.log")

        self.handle = None

    def get_cmdline(self) -> str:
        return f"python3 -m vllm.entrypoints.openai.api_server {self.config.vllm_config.cmdargs()} {self.config.vllm_optional_config.cmdargs()} {self.config.lmcache_config.cmdargs()}"

    def start(self):
        self.handle = run_command(self.command, self.stdout_log, self.stderr_log, detach=True, **self.config.envs)
        pass

    def wait_until_ready(self) -> bool:
        # Try reading the log file to see if the server is ready
        while True:
            if not self.is_healthy():
                return False

            if not os.path.exists(self.stdout_log):
                return False

            try:
                output = subprocess.check_output(
                        f"grep 'Uvicorn running on http' {self.stdout_log} {self.stderr_log} | wc -l",
                        shell=True).decode().strip()
                output = int(output)
                logger.debug(f"LocalVllmBootstrapper::wait_until_ready: output {output}")
                if output > 0:
                    return True
            except Exception as e:
                logger.error(f"LocalVllmBootstrapper::wait_until_ready: Error reading log file{e}")
                return False

            time.sleep(1)

    def is_healthy(self) -> bool:
        if self.handle is not None:
            return self.handle.is_alive()
        return False

    def close(self):
        logger.info("Closing LocalVllmBootstrapper")
        if hasattr(self, "handle") and self.handle is not None:
            self.handle.kill_and_close()

#if __name__ == "__main__":
#    config = BootstrapConfig(
#        vllm_config=VLLMConfig(
#            port = 8000,
#            model = "mistralai/Mistral-7B-Instruct-v0.2",
#            gpu_memory_utilization = 0.5,
#            tensor_parallel_size = 1),
#        vllm_optional_config = VLLMOptionalConfig(),
#        lmcache_config = LMCacheConfig("configs/example-local.yaml"),
#        envs = {"CUDA_VISIBLE_DEVICES": "0"})
#
#    engine = LocalVllmBootstrapper(config)
#    engine.start()
#    print(engine.is_healthy())
#    print(engine.wait_until_ready())
#    engine.close()

