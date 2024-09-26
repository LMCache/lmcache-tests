# End-to-end test for LMCache

> Note: currently, this doc is for onboarding the new developers. Will have a separate README in the future for general audiences.

It's recommended to create a new folder before cloning the repository. The final file structure will look like as follows:

```
<parent-folder>/
©À©¤©¤ lmcache-test/
©À©¤©¤ LMCache/
©¸©¤©¤ lmcache-vllm/
```

## 1. Environment installation


```bash
# Create conda environment
conda create -n lmcache python=3.10
conda activate lmcache

# Clone github repository
git clone git@github.com:LMCache/lmcache-tests.git
cd lmcache-tests

# Run the installation script
bash prepare_environment.sh
```

## 2. Run the tests

### 2.1 Quickstart example

The following command line runs the test `test_lmcache_local_cpu` defined in `tests/tests.py` and write the output results to the output folder (`outputs/test_lmcache_local_cpu.csv`).

```bash
python3 main.py tests/tests.py -f test_lmcache_local_cpu -o outputs/
```

To process the result, please run
```bash
cd outputs/
python3 process_result.py
```
Then, a pdf file `test_lmcache_local_cpu.pdf` will be created.

You can also monitor the following files to check the status of the bootstrapped vllm process.

For stderr:
```bash
tail -f /tmp/8000-stderr.log
```

For stdout:
```bash
tail -f /tmp/8000-stdout.log
```


### 2.2 Usage of main.py

`main.py` is the entrypoint to execute the test functions:
```
usage: main.py [-h] [-f FILTER] [-l] [-o OUTPUT_DIR] filepath

Execute all functions in a given Python file.

positional arguments:
  filepath              The Python file to execute functions from (include subfolders if any).

options:
  -h, --help            show this help message and exit
  -f FILTER, --filter FILTER
                        Pattern to filter which functions to execute.
  -l, --list            List all functions in the module without executing them.
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        The directory to put the output file.
```

Here are some examples:

```bash
# Run all the test functions defined in 'tests/tests.py' and save the output to 'outputs/'
python3 main.py tests/tests.py -o outputs/

# List the tests in 'tests/tests.py'
python3 main.py tests/tests.py -l

# Run some specific tests that match the given pattern (e.g., containing 'cachegen')
python3 main.py tests/tests.py -f cachegen
```

### 2.3 Output parsing

In general, each test function should output the results as a csv file, where the file name is the same as function name but with a `csv` suffix.
There should be multiple columns in the CSV:
- `expr_id`: the id of the experiment
- `timestamp`: the timestamp of the request in the workload
- `engine_id`: the id of the serving engine instance to which the request is sent
- `request_id`: the id of the request in the workload
- `TTFT`: the time-to-first-token of this request
- `throughput`: the number of output tokens per second of this request
- `context_len`: the context length of this request
- `query_len`: the query length of this request
- `gpu_memory`: the gpu memory used before this experiment is finished

Some example codes of how to parse the output CSV can be found in `outputs/process_result.py`.


## 3. Contributing guide: understanding the code

### 3.1 Basic terminology

- **`Request`**: A request is a single prompt containing some context and a user query.
- **`Engine`**: Means "serving engine". An engine is an LLM serving engine process that can process requests through the OpenAI API interface.
- **`Workload`**: A workload is a list of requests at different timestamps. Usually, a workload is associated with only one engine.
- **`Experiment`**: An experiment runs multiple engines simultaneously and sends the workloads generated from the _same_ configuration to the serving engines.
- **`Test case`**: A test case contains a list of experiments. The goal is to compare the performance of different engines under different workloads. 
- **`Test function`**: A test function wraps the test cases and saves the output CSV. In most cases, a single test function only contains one test case. Different test functions aim to test different functionalities of the system.

### 3.2 Main components

**Test case configuration**:

Test case configuration controls the experiments to run. The configuration-related code can be found in [config.py](https://github.com/LMCache/lmcache-tests/blob/main/configs.py).

Currently, we support the following configurations:
- **Workload configuration**: Specifies the QPS, context length, query length, and total duration of the workload.
- **vLLM configuration**: Controls the command line arguments used to start vLLM.
- **LMCache configuration**: Points to a LMCache configuration file.

During the experiment, workload configuration will be used to generate the workloads, and vLLM configuration + LMCache configuration will be used to start the engine.

**Workload generator**:

The workload generator takes in a workload configuration and generates the workload (*i.e.*, a list of requests at different timestamps) as the output.
The code for the workload generator can be found in [workload.py](https://github.com/LMCache/lmcache-tests/blob/main/workload.py).

By design, there could be multiple different kinds of workload generators for different use cases, such as chatbot, QA, or RAG. 
The class `Usecase` is used to specify which workload generator to create during runtime. 
Currently, we only support a `DUMMY` use case where the requests in the generated workload only contain dummy texts and questions.

The workload generator, once initialized with a configuration, only provides a single method: `generate(self) -> Workload`.

**Engine bootstrapper**:

The engine bootstrapper pulls up the serving engine based on the configurations (vLLM configuration + LMCache configuration).
Currently, we only support starting vLLM (with or without LMCache) from the terminal. We will support docker-based engines in the future.
The code can be found in [bootstrapper.py](https://github.com/LMCache/lmcache-tests/blob/main/bootstrapper.py)

The engine bootstrapper supports the following methods:
- `start(self)`: Non-blocking function to start the serving engine.
- `wait_until_ready(self, timeout=60) -> bool`: Block until the serving engine is ready or it's dead or timeout is reached. Returns true if it's ready, otherwise false.
- `is_healthy(self) -> bool`: Non-block function to check if the engine is alive or not.
- `close(self)`: blocking function to shutdown the serving engine

**Experiment runner**:

The experiment runner takes in _one_ workload config and $N$ engine configs as input. It does the following things:
- Create the $N$ workload generators from the workload config.
- Generate $N$ workloads.
- Bootstrap all the serving engines. (This requires that the engine config is correct in a way that multiple serving engines can run at the same time).
- Send the requests at the given timestamps to the engines.
- Collect the TTFT, throughput, and the GPU memory usage.
- Return the results in a pandas dataframe.

The code can be found in [driver.py](https://github.com/LMCache/lmcache-tests/blob/main/driver.py)

## 4. Contributing guide: adding new tests

(WIP)
