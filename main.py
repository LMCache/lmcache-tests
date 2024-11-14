import argparse
import importlib
import sys
import re
import os
import inspect
import pandas as pd

from log import init_logger
logger = init_logger(__name__)

def wrapped_test_runner(test_func, model, output_file):
    """
    Run the test case and save results to output file
    """
    if model is None:
        output_df = test_func()
    else:
        output_df = test_func(model)
    if isinstance(output_df, pd.DataFrame):
        output_df.to_csv(output_file, index=False)
        logger.info("Saving output to " + output_file)
    else:
        logger.warning(f"Output is {type(output_df)}, not a DataFrame. Skipping saving to file.")

def main():
    
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    
    parser = argparse.ArgumentParser(description="Execute all functions in a given Python file.")
    parser.add_argument("filepath", help="The Python file to execute functions from (include subfolders if any).")
    parser.add_argument("-f", "--filter", help="Pattern to filter which functions to execute.", default=None)
    parser.add_argument("-l", "--list", help="List all functions in the module without executing them.", action="store_true")
    parser.add_argument("-o", "--output-dir", help="The directory to put the output file.", default="outputs/")
    parser.add_argument("-m", "--model", help="The models of vllm for every functions.", default=None)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Adjust the Python path to include the directory containing the module
    module_path, module_file = os.path.split(args.filepath)
    module_name = module_file.replace('.py', '')

    if module_path not in sys.path:
        sys.path.insert(0, module_path)

    # Import the module dynamically
    try:
        logger.info(f"Importing testing script {module_name}...")
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        logger.error(f"Error: The testing script {module_name} could not be found.")
        sys.exit(1)
    except ImportError as e:
        print(f"Import Error: {e}")
        sys.exit(1)

    # Filter functions in the module
    name_filter = "test_"
    if args.filter:
        logger.info(f"Filtering functions with pattern: {args.filter}")
        name_filter = args.filter

    function_list = []
    function_names = []
    for attr_name, attr_value in inspect.getmembers(module, inspect.isfunction):
        if attr_value.__module__ == module.__name__ and callable(getattr(module, attr_name)):
            func = getattr(module, attr_name)
            if re.search(name_filter, attr_name):
                logger.info(f"Found function: {attr_name}")
                function_list.append(func)
                function_names.append(attr_name)

    logger.info(f"Collected {len(function_list)} functions to execute.")

    if args.list:
        logger.info("Listing functions in the module:")
        for name in function_names:
            print(name)
        sys.exit(0)

    if args.model is None:
        for func, name in zip(function_list, function_names):
            logger.info("Executing function: " + name)
            wrapped_test_runner(func, None, f"outputs/{name}.csv")
    else:
        model_prefix = args.model.split('/')[0]
        for func, name in zip(function_list, function_names):
            logger.info("Executing function: " + name)
            wrapped_test_runner(func, args.model, f"outputs/{name}_{model_prefix}.csv")
    
if __name__ == "__main__":
    main()

