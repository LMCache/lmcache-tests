# End-to-end test for LMCache

WIP...


## Run the tests

```bash
# Run all the test functions defined in 'tests/tests.py' and save the output to 'outputs/'
python3 main.py tests/tests.py -o outputs/

# List the tests in 'tests/tests.py'
python3 main.py tests/tests.py -l

# Run some specific tests that matches the given pattern (e.g., containing 'cachegen')
python3 main.py tests/tests.py -f cachegen
```

## Parsing the results

See `outputs/process_result.py`
