# End-to-end test for LMCache

## 1. Environment installation

```bash
# Create conda environment
conda create -n lmcache python=3.10
conda activate lmcache

# run the installation script
bash prepare_environment.sh
```

## 2. Run the tests

### 2.1 Quick start example

The following commandline runs the test `test_lmcache_local_cpu` defined in `tests/tests.py` and write the output results to the output folder (`outputs/test_lmcache_local_cpu.csv`).

```bash
python3 main.py tests/test.py -f test_lmcache_local_cpu -o outputs/
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

# Run some specific tests that matches the given pattern (e.g., containing 'cachegen')
python3 main.py tests/tests.py -f cachegen
```

### 2.3 Output parsing

In general, each test function should output the results as a csv file, where the file name is the same as function name but with a `csv` suffix.
There should be multiple columns in the csv:
- `expr_id`: the id of the experiment
- `timestamp`: the timestamp of the request in the workload
- `engine_id`: the id of the serving engine instance to which the request is sent
- `request_id`: the id of the request in the workload
- `TTFT`: the time-to-first-token of this request
- `throughput`: the number of output tokens per second of this request
- `context_len`: the context length of this request
- `query_len`: the query length of this request
- `gpu_memory`: the gpu memory used before this experiment is finished

Some example code of how to parse the output csv can be found in `outputs/process_result.py`.


## 3. Contributing guide: understanding the code

### Basic terminology

### Main components

### Key interfaces


## 4. Contributing guide: adding new tests
