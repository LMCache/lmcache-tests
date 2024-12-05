from typing import List, Tuple
import pandas as pd
import time
import multiprocessing
from dataclasses import dataclass
import openai

from configs import BootstrapConfig, WorkloadConfig, Usecase
from test_cases import TestCase
from bootstrapper import CreateBootstrapper, Bootstrapper, LMCacheServerManager
from workload import CreateWorkloadGenerator, Request, MultiTurnWorkloadGenerator
from utils import read_gpu_memory, get_max_context_length

import log
logger = log.init_logger(__name__)

@dataclass
class ExperimentResult:
    timestamp: str
    engine_id: int
    request_id: int
    TTFT: float
    throughput: float

@dataclass
class ExperimentResultWithOutput:
    timestamp: str
    engine_id: int
    request_id: int
    TTFT: float
    throughput: float
    messages: str

class RequestExecutor:
    def __init__(self):
        pass
    
    def schedule_requests(self, requests_list: List[List[Request]], clients: List[openai.Client], models: List[str]):
        """
        Take in the list of requests and the clients, prepare the execution 
        """
        requests_list = [(cid, rid, req, client, model) \
                for cid, (requests, client, model) in enumerate(zip(requests_list, clients, models))\
                for rid, req in enumerate(requests)]

        # Order the requests by the timestamps
        self.pending_requests = sorted(requests_list, key=lambda x: x[2].timestamp)

    def execute_one_request(
            self, 
            client_id: int, 
            request_id: int, 
            request: Request, 
            client: openai.Client, 
            model: str,
            queue: multiprocessing.Queue) -> Tuple[float, float]:
        """
        Execute the request and put the result into the queue
        """
        ttft, thp = execute_openai_request(request, model, client)
        logger.info(f"Request completed, TTFT = {ttft}, throughput = {thp}")
        queue.put(ExperimentResult(request.timestamp, client_id, request_id, ttft, thp))

    def execute_one_request_with_output(
            self, 
            client_id: int, 
            request_id: int, 
            request: Request, 
            client: openai.Client, 
            model: str,
            queue: multiprocessing.Queue) -> Tuple[float, float]:
        """
        Execute the request and put the result into the queue
        """
        ttft, thp, messages = execute_openai_request_with_output(request, model, client)
        logger.info(f"Request completed, TTFT = {ttft}, throughput = {thp}")
        queue.put(ExperimentResultWithOutput(request.timestamp, client_id, request_id, ttft, thp, messages))

    def execute_all(self) -> List[ExperimentResult]:
        """
        Returns the list of expr results
        """
        queue = multiprocessing.Queue()
        start_time = time.time()
        processes = []
        try:
            for client_id, request_id, request, client, model in self.pending_requests:
                already_elapsed = time.time() - start_time
                # Wait for the request to be ready
                wait_time = request.timestamp - already_elapsed
                if wait_time > 0:
                    time.sleep(wait_time)
                # Execute the request by a new process
                process = multiprocessing.Process(
                        target = self.execute_one_request, 
                        args=(client_id, request_id, request, client, model, queue))
                process.start()
                processes.append(process)

            # Wait for all the processes to finish
            for process in processes:
                process.join()

        except Exception as e:
            logger.error(f"Exception happend when sending request: {e}")
            for process in processes:
                process.terminate()
            return []

        return [queue.get() for _ in self.pending_requests]
    
    def execute_all_with_output(self) -> List[ExperimentResultWithOutput]: 
        """
        Returns the list of expr results
        """
        queue = multiprocessing.Queue()
        start_time = time.time()
        processes = []
        try:
            for client_id, request_id, request, client, model in self.pending_requests:
                already_elapsed = time.time() - start_time
                # Wait for the request to be ready
                wait_time = request.timestamp - already_elapsed
                if wait_time > 0:
                    time.sleep(wait_time)
                # Execute the request by a new process
                process = multiprocessing.Process(
                        target = self.execute_one_request_with_output, 
                        args=(client_id, request_id, request, client, model, queue))
                process.start()
                processes.append(process)

            # Wait for all the processes to finish
            for process in processes:
                process.join()

        except Exception as e:
            logger.error(f"Exception happend when sending request: {e}")
            for process in processes:
                process.terminate()
            return []

        return [queue.get() for _ in self.pending_requests]


def create_openai_client(port: int, model) -> openai.Client:
    openai_api_key = "EMPTY"
    # TODO: currently we assume the engines are open to localhost. Need to support different hostname in the future
    openai_api_base = f"http://localhost:{port}/v1"

    client = openai.OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    messages = [{
        "role": "user",
        "content": f"This is a warm up request" * 200
        }]

    chat_completion = client.chat.completions.create(
        messages = messages,
        model = model,
        temperature = 0,
        stream = False,
        max_tokens = 1,
    )

    return client


def execute_openai_request(request: Request, model: str, client: openai.Client) -> Tuple[float, float]:
    """
    Execute a single request to the OpenAI engine
    Returns: TTFT (seconds) and throughput (tokens per second)
    """

    messages = [{
        "role": "user",
        "content": f"{request.context} {request.question}"
        }]

    #import random
    #t = random.randint(2, 8)
    #time.sleep(t)
    #return t, t

    
    try:
        logger.debug("Issusing a new request...")
        chat_completion = client.chat.completions.create(
                messages = messages,
                model = model,
                temperature = 0,
                stream = True,
                max_tokens = 200,
            )

        start_time = time.perf_counter()
        first_token_time = None
        ntokens = 0
        messages = []
        for chunk in chat_completion:
            chunk_message = chunk.choices[0].delta.content
            if chunk_message is not None:
                if first_token_time is None and chunk_message != " " and chunk_message != "":
                    first_token_time = time.perf_counter()
                messages.append(chunk_message)
                ntokens += 1
        end_time = time.perf_counter()

        ttft = first_token_time - start_time
        throughput = ntokens / (end_time - first_token_time)
        logger.debug(f"Response: {''.join(messages)}")
    except Exception as e:
        logger.error(f"OpenAI request failed: {e}")
        return -1, -1

    return ttft, throughput

def execute_openai_request_with_output(request: Request, model: str, client: openai.Client) -> Tuple[float, float, str]:
    """
    Execute a single request to the OpenAI engine
    Returns: TTFT (seconds) and throughput (tokens per second)
    """
    messages = [
        {
            "role": "user",
            "content": request.question
        }
    ]
    split_strings = request.context.split("<<splitter>>")
    for split_string in split_strings:
        if split_string != "":
            messages.extend([
                {
                    "role": "assistant",
                    "content": split_string
                },
                {
                    "role": "user",
                    "content": request.question
                }
            ])
    try:
        logger.debug("Issusing a new request...")
        chat_completion = client.chat.completions.create(
                messages = messages,
                model = model,
                temperature = 0,
                stream = True,
                max_tokens = 2000,
            )

        start_time = time.perf_counter()
        first_token_time = None
        ntokens = 0
        messages = []
        for chunk in chat_completion:
            chunk_message = chunk.choices[0].delta.content
            if chunk_message is not None:
                if first_token_time is None and chunk_message != " " and chunk_message != "":
                    first_token_time = time.perf_counter()
                messages.append(chunk_message)
                ntokens += 1
        end_time = time.perf_counter()

        ttft = first_token_time - start_time
        throughput = ntokens / (end_time - first_token_time)
        logger.debug(f"Response: {''.join(messages)}")
    except Exception as e:
        logger.error(f"OpenAI request failed: {e}")
        return -1, -1

    return ttft, throughput, f"{''.join(messages)}"

def run_experiment(
        workload_config: WorkloadConfig, 
        usecase: Usecase, 
        engine_configs: List[BootstrapConfig]) -> List[ExperimentResult]:
    """
    Run a single experiment: 
    - Bootstrap the serving bootstrappers
    - Generate workload for each engine
    - Wait for engine ready
    - (separate threads/processes) Send requests, measure TTFT and throughput
    - (separate threads/processes) Measure GPU stats
    - Close the bootstrappers

    Returns:
        - None if the experiment failed
        - <workload desc> <timestamp> <engine id> <request id> <TTFT> <throughput> <GPU mem util>
    """
    def cleanup(bootstrappers: List[Bootstrapper]):
        logger.info("Cleanning up the engine processes")
        for bootstrapper in bootstrappers:
            bootstrapper.close()
        LMCacheServerManager.close_servers()

    logger.info(f"Running experiment: {workload_config.desc()} {usecase}")

    # Create the workloads
    workload_generators = []
    for engine_config in engine_configs:
        workload_generators.append(CreateWorkloadGenerator(workload_config, usecase, get_max_context_length(engine_config.vllm_config.model)))
    workloads = [generator.generate() for generator in workload_generators]


    # Start the serving engine
    bootstrappers = []
    for config in engine_configs:
        bootstrappers.append(CreateBootstrapper(config))
        bootstrappers[-1].start()

    try:
        # Wait for the engines to be ready
        for bootstrapper in bootstrappers:
            ready = bootstrapper.wait_until_ready(timeout = 300)
            if not ready:
                logger.error(f"Engine {bootstrapper} is not ready")
                cleanup(bootstrappers)
                return

        # Create the clients
        clients = [create_openai_client(config.vllm_config.port, config.vllm_config.model) for config in engine_configs]
        models = [config.vllm_config.model for config in engine_configs]

        # Execute the requests
        executor = RequestExecutor()
        executor.schedule_requests(workloads, clients, models)
        results = executor.execute_all()

        #print(results)

        # Read GPU memory utilization
        gpu_usage = read_gpu_memory()

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        cleanup(bootstrappers)
        return None

    finally:
        # Cleanup
        cleanup(bootstrappers)

    return results, gpu_usage

def run_multi_turn_experiment(
        workload_config: WorkloadConfig, 
        usecase: Usecase, 
        engine_configs: List[BootstrapConfig]) -> List[ExperimentResultWithOutput]:
    """
    Run a multi turn experiment: 
    - Bootstrap the serving bootstrappers
    - Create workload generator for each engine
    - Wait for engine ready
    - (loop) Generate workload and send requests
    - Close the bootstrappers

    Returns:
        - None if the experiment failed
        - <workload desc> <timestamp> <engine id> <request id> <TTFT> <throughput> <GPU mem util>
    """
    def cleanup(bootstrappers: List[Bootstrapper]):
        logger.info("Cleanning up the engine processes")
        for bootstrapper in bootstrappers:
            bootstrapper.close()
        LMCacheServerManager.close_servers()

    logger.info(f"Running experiment: {workload_config.desc()} {usecase}")

    # Create the workloads
    workload_generators = [MultiTurnWorkloadGenerator(workload_config) for _ in engine_configs]

    # Start the serving engine
    bootstrappers = []
    for config in engine_configs:
        bootstrappers.append(CreateBootstrapper(config))
        bootstrappers[-1].start()

    try:
        # Wait for the engines to be ready
        for bootstrapper in bootstrappers:
            ready = bootstrapper.wait_until_ready(timeout = 300)
            if not ready:
                logger.error(f"Engine {bootstrapper} is not ready")
                cleanup(bootstrappers)
                return

        # Create the clients
        clients = [create_openai_client(config.vllm_config.port, config.vllm_config.model) for config in engine_configs]
        models = [config.vllm_config.model for config in engine_configs]

        # Execute the requests
        executor = RequestExecutor()

        # Generation and execution in multi turns (hard-coded 10 turns)
        results = []
        gpu_usage = []
        for i in range(10):
            workloads = [generator.generate() for generator in workload_generators]
            executor.schedule_requests(workloads, clients, models)
            results.append(executor.execute_all_with_output())
            [setattr(result, 'request_id', i) for result in results[i]]
            gpu_usage.append(read_gpu_memory())
            for result in results[i]:
                workload_generators[result.engine_id].offset += 1 / workload_config.qps
                workload_generators[result.engine_id].store(result.messages)

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        cleanup(bootstrappers)
        return None

    finally:
        # Cleanup
        cleanup(bootstrappers)

    return results, gpu_usage

def run_test_case(case: TestCase) -> pd.DataFrame:
    """
    Run a single test case
    """
    dfs = []
    for expr_id, (workload_cfg, usecase) in enumerate(case.experiments):
        if usecase == Usecase.MULTI:
            results = run_multi_turn_experiment(workload_cfg, usecase, case.engines)
        else:
            results = run_experiment(workload_cfg, usecase, case.engines)
        if results is None:
            logger.error(f"Experiment failed: {workload_cfg.desc()} {usecase}")
            continue
        else:
            results, gpu_usage = results

        if usecase == Usecase.MULTI:
            for idx, result in enumerate(results):
                dataframe = pd.DataFrame([item.__dict__ for item in result])
                dataframe = dataframe.drop(columns=['messages'])
                dataframe = dataframe.sort_values(by=["timestamp", "engine_id", "request_id"])
                dataframe["context_len"] = workload_cfg.context_length
                dataframe["query_len"] = workload_cfg.query_length
                #dataframe["workload"] = workload_cfg.desc()
                dataframe["gpu_memory"] = gpu_usage[idx]
                dataframe["expr_id"] = "-"
                dfs.append(dataframe)
        else:
            dataframe = pd.DataFrame([item.__dict__ for item in results])
            dataframe = dataframe.sort_values(by=["timestamp", "engine_id", "request_id"])
            # TODO: need to handle the case when max context length is reached
            if usecase == Usecase.VARY:
                dataframe["context_len"] = [workload_cfg.context_length * (idx // 2 + 1) for idx in range(1, len(dataframe)+1)]
            else:
                dataframe["context_len"] = workload_cfg.context_length
            dataframe["query_len"] = workload_cfg.query_length
            #dataframe["workload"] = workload_cfg.desc()
            dataframe["gpu_memory"] = gpu_usage
            dataframe["expr_id"] = expr_id
            dfs.append(dataframe)
    return pd.concat(dfs)

def run_test_cases(cases: List[TestCase]) -> pd.DataFrame:
    """
    Run multiple test cases
    The returned dataframe will have a new column "case_id"
    """
    dataframes = []
    for case_id, case in enumerate(cases):
        dataframe = run_test_case(case)
        dataframe["case_id"] = case_id
        dataframes.append(dataframe)

    return pd.concat(dataframes)
    

#if __name__ == "__main__":
#    # test the request executor
#    #requests = [
#    #        [Request(1, "context1", "question1"), Request(3, "context1", "question2")],
#    #        [Request(2, "context2", "question1"), Request(4, "context2", "question2")],
#    #    ]
#    #clients = ["client1", "client2"]
#    #executor = RequestExecutor("model_name")
#    #executor.schedule_requests(requests, clients)
#    #print(executor.pending_requests)
#    #for val in executor.execute_all():
#    #    print(val)
#
#    from configs import VLLMConfig, VLLMOptionalConfig, LMCacheConfig, EngineType
#    vllm_config1 = VLLMConfig(
#        port = 8000,
#        model = "mistralai/Mistral-7B-Instruct-v0.2",
#        gpu_memory_utilization = 0.5,
#        tensor_parallel_size = 1)
#
#    vllm_config2 = VLLMConfig(
#        port = 8001,
#        model = "mistralai/Mistral-7B-Instruct-v0.2",
#        gpu_memory_utilization = 0.5,
#        tensor_parallel_size = 1)
#
#    config = BootstrapConfig(
#        engine_type = EngineType.LOCAL,
#        vllm_config = vllm_config1,
#        vllm_optional_config = VLLMOptionalConfig(),
#        lmcache_config = LMCacheConfig("configs/example.yaml"),
#        #lmcache_config = LMCacheConfig(None),
#        envs = {"CUDA_VISIBLE_DEVICES": "0"})
#
#    import copy
#    config2 = copy.deepcopy(config)
#    config2.vllm_config = vllm_config2
#    config2.envs = {"CUDA_VISIBLE_DEVICES": "1"}
#
#    workload_config = WorkloadConfig(1, 3, 1000, 100)
#
#    test_case = TestCase([(workload_config, Usecase.DUMMY)], [config, config2])
#    #run_experiment(workload_config, Usecase.DUMMY, [config, config2])
#    results = run_test_case(test_case)
#    print(results)
