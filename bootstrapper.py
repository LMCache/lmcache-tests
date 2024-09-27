import abc
import re
import yaml
import time
import subprocess
import os
from typing import Tuple, Optional

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
    def wait_until_ready(self, timeout=60) -> bool:
        """
        Block until the serving engine is ready or it's dead or timeout is reached
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


    def _monitor_file_output(self, filenames, pattern, timeout = 60):
        """
        Monitors the output of the files in filenames for the pattern
        Also check if is_healthy or not
        Returns: 
            True if the pattern is found in any of the files or it becomes unhealthy
            False otherwise
        """
        if not self.is_healthy():
            return False

        start_time = time.time()
        while time.time() - start_time < timeout:
            for filename in filenames:
                if not os.path.exists(filename):
                    continue

                with open(filename, 'r') as fin:
                    for line in fin:
                        if re.search(pattern, line):
                            return True

            time.sleep(1)
            logger.debug(f"Waiting for the pattern '{pattern}' in the files {filenames}")

        logger.error(f"Timeout waiting for the pattern '{pattern}' in the files {filenames}")
        return False

def CreateBootstrapper(config: BootstrapConfig, log_dir = "/tmp") -> Bootstrapper:
    match config.engine_type:
        case EngineType.LOCAL:
            return LocalVllmBootstrapper(config, log_dir)
        case _:
            raise ValueError(f"Unsupported engine type: {config.engine_type}")

class LocalLMCacheServerBootstrapper(Bootstrapper):
    """
    Bootstraps a local lmcache server
    """
    def __init__(self, config: BootstrapConfig, log_dir = "/tmp"):
        super().__init__(config)
        server_config = self.parse_lmcache_server_config(self.config.lmcache_config.config_path)
        self.handle = None
        self.started = False

        match config.lmcache_config.remote_device:
            case None:
                self.remote_device = "cpu"
            case path:
                self.remote_device = path

        if server_config is None:
            self.is_needed = False
        else:
            self.is_needed = True
            self.host, self.port = server_config
            self.user_name=os.popen('whoami').read()[:-1]
            self.stdout_log = os.path.join(log_dir, f"{self.user_name}-lmcache-server-{self.port}-stdout.log")
            self.stderr_log = os.path.join(log_dir, f"{self.user_name}-lmcache-server-{self.port}-stderr.log")

    def parse_lmcache_server_config(self, config_file: str) -> Optional[Tuple[str, int]]:
        """
        Returns the lmcache server host and port if it's lm://<host>:<port>
        Otherwise None
        """
        if config_file is None:
            return None

        with open(config_file, 'r') as fin:
            config = yaml.safe_load(fin)

        remote_url = config.get("remote_url", None)
        if remote_url is None:
            return None

        m = re.match(r"(.*)://(.*):(\d+)", remote_url)
        connector_type, host, port = m.group(1), m.group(2), int(m.group(3))
        if connector_type == "lm":
            return host, port
        else:
            return None

    def start(self):
        if self.started:
            return

        if not self.is_needed:
            return

        # cmd = f"python3 -um lmcache_server.server {self.host} {self.port} {self.remote_device}"
        cmd = f"python -um lmcache.server  {self.host} {self.port} "
        print(f"\033[32mLaunching Remote LMCache Server with Command :\033[0m {cmd}")
        self.handle = run_command(cmd, self.stdout_log, self.stderr_log, detach=True)
        self.started = True

    def wait_until_ready(self, timeout = 60) -> bool:
        if not self.is_needed:
            # Always true if not needed
            return True

        if not self.is_healthy():
            logger.error("LMCacheServer is dead!")
            return False

        return self._monitor_file_output([self.stdout_log, self.stderr_log], "Server started at", timeout=timeout)

        return True

    def is_healthy(self) -> bool:
        if not self.is_needed:
            # Always true if not needed
            return True

        if self.handle is not None:
            return self.handle.is_alive()

        return False

    def close(self):
        logger.info("Closing LocalLMCacheServer")
        if hasattr(self, "handle") and self.handle is not None:
            self.handle.kill_and_close(force_kill_after = 3)

class LMCacheServerManager:
    _instances = {}
    _engine_types = {}

    @classmethod
    def get_or_create(
            cls,
            config: BootstrapConfig,
        ) -> Bootstrapper:
        """
        Creates or return the existing instance of the LMCacheServer.
        Key is the configuration and the value is the instance
        """
        instance_id = config.lmcache_config.config_path

        if instance_id not in cls._instances:
            match config.engine_type:
                case EngineType.LOCAL:
                    cls._instances[instance_id] = LocalLMCacheServerBootstrapper(config)
                case _:
                    raise ValueError(f"Unsupported engine type: {config.engine_type}")
            cls._engine_types[instance_id] = config.engine_type
        else:
            if cls._engine_types[instance_id] != config.engine_type:
                raise ValueError(f"Trying to create a new instance with different engine type {config.engine_type} for the same configuration {instance_id}")

        return cls._instances[instance_id]

    @classmethod
    def close_servers(cls):
        """
        Close and remove all the active lmcache servers
        """
        for instance_id, instance in cls._instances.items():
            instance.close()
        cls._instances = {}
        cls._engine_types = {}


                        
class LocalVllmBootstrapper(Bootstrapper):
    """
    Bootstraps a local vllm server
    """
    def __init__(self, config: BootstrapConfig, log_dir = "/tmp"):
        super().__init__(config)
        self.command = self.get_cmdline()
        self.user_name=os.popen('whoami').read()[:-1]
        self.stdout_log = os.path.join(log_dir, f"{self.user_name}-{self.config.vllm_config.port}-stdout.log")
        self.stderr_log = os.path.join(log_dir, f"{self.user_name}-{self.config.vllm_config.port}-stderr.log")

        self.handle = None
        self.lmcache_server_handler = LMCacheServerManager.get_or_create(config)

    def get_cmdline(self) -> str:
        extra_args = "--trust-remote-code"
        if self.config.lmcache_config.cmdargs() == " ":
            os.environ["LMCACHE_CONFIG_FILE"]=self.config.lmcache_config.config_path
            print(f"\033[32mLaunching Engine with Command :\033[0m LMCACHE_CONFIG_FILE={self.config.lmcache_config.config_path}  lmcache_vllm serve {self.config.vllm_config.cmdargs()} {self.config.vllm_optional_config.cmdargs()} {extra_args}")
            return f"lmcache_vllm serve {self.config.vllm_config.cmdargs()} {self.config.vllm_optional_config.cmdargs()} {extra_args}"
        else:
            print(f"\033[32mLaunching Engine with Command :\033[0m vllm serve {self.config.vllm_config.cmdargs()} {self.config.vllm_optional_config.cmdargs()} {extra_args}")
            return f"vllm serve {self.config.vllm_config.cmdargs()} {self.config.vllm_optional_config.cmdargs()} {extra_args}"


        # return f"python3 -m vllm.entrypoints.openai.api_server {self.config.vllm_config.cmdargs()} {self.config.vllm_optional_config.cmdargs()} {self.config.lmcache_config.cmdargs()} {extra_args}"

    def start(self):
        self.lmcache_server_handler.start()
        self.lmcache_server_handler.wait_until_ready(timeout = 10)

        self.handle = run_command(
                self.command, 
                self.stdout_log, self.stderr_log, 
                detach=True, **self.config.envs)

    def wait_until_ready(self, timeout = 60) -> bool:
        # Try reading the log file to see if the server is ready
        if not self.is_healthy():
            logger.error(f"VLLM or lmcache server is dead!")
            return False

        if not os.path.exists(self.stdout_log):
            return False

        return self._monitor_file_output([self.stdout_log, self.stderr_log], "Uvicorn running on http", timeout=timeout)


    def is_healthy(self) -> bool:
        if not self.lmcache_server_handler.is_healthy():
            logger.warn(f"LMCache Server is dead during vLLM's check!")
            return False 

        if self.handle is not None:
            return self.handle.is_alive()
        return False

    def close(self):
        logger.info("Closing LocalVllmBootstrapper")
        if hasattr(self, "handle") and self.handle is not None:
            self.handle.kill_and_close(force_kill_after = 5)

        if hasattr(self, "lmcache_server_handler") and self.lmcache_server_handler is not None:
            self.lmcache_server_handler.close()

#if __name__ == "__main__":
#    config = BootstrapConfig(
#        engine_type = EngineType.LOCAL,
#        vllm_config=VLLMConfig(
#            port = 8000,
#            model = "mistralai/Mistral-7B-Instruct-v0.2",
#            gpu_memory_utilization = 0.5,
#            tensor_parallel_size = 1),
#        vllm_optional_config = VLLMOptionalConfig(),
#        #lmcache_config = LMCacheConfig("configs/example-local.yaml"),
#        lmcache_config = LMCacheConfig("configs/example.yaml"),
#        envs = {"CUDA_VISIBLE_DEVICES": "0"})
#
#    #engine = LMCacheServerManager.get_or_create(config)
#    #engine.start()
#    #print(engine.is_healthy())
#    #print(engine.wait_until_ready(timeout = 10))
#    #engine.close()
#    engine = LocalVllmBootstrapper(config)
#    engine.start()
#    print(engine.is_healthy())
#    print(engine.wait_until_ready(timeout=120))
#    engine.close()
#
