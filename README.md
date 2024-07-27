# lmcache-tests
## Quick installation (LMCache, lmcache-vllm, lmcache-server, vllm)
### Run `bash one_install.sh` under your desired directory 

## Running the tests
### Step 1 (optional): config lmcache and vllm under `configs/`

### Step 2: setup cache server and vllm nodes
```
python setup.py --node all
```

### Step 3: Run tests
In a different terminal:
```
python test_performance.py
```
The above test reproduces TTFT (full computation) vs. TTFT (loading cache from remote cpu).

## Benchmark results
### E2E results (TTFT)
| Model  | Context Length |Hardware settings | Original | Original w store| Local CPU| Remote CPU | Remote CPU (Pipelined)|
| ------------- | ------------- |------------- |------------- |------------- |------------- |------------- |------------- |
| mistralai/Mistral-7B-Instruct-v0.2 | 25503 | 1 * A40 | 5.259| 5.883-6.807| 0.2517| 1.722-1.967| 1.395-1.491|
| THUDM/glm-4-9b-chat  | 20897 |1 * A40  |4.820| | | | 0.661-0.717|

### Performance breakdown