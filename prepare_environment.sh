#!/bin/bash
set -e

pip install matplotlib


# Check conda
if [[ -z $CONDA_DEFAULT_ENV || $CONDA_DEFAULT_ENV == "base" ]]; then
    echo -e "\033[31mPlease activate a conda environment before running this script.\033[0m"
    echo -e "\033[31mRun \`conda create -n lmcache python=3.10\` then \`conda activate lmcache\`.\033[0m"
    exit 1
fi

# Check python version matches 3.10
python_version=$(python3 -c "import sys; print(sys.version_info[:2])")
if [[ ! "$python_version" =~ ^\(3,\ 1[0-9]\) ]]; then
    echo -e "\033[31mPlease use Python >= 3.10 to run this script.\033[0m"
    exit 1
fi

# Step 1: clone the code
pip install vllm==0.6.2.3
git clone https://github.com/LMCache/LMCache.git ../LMCache
git clone https://github.com/LMCache/lmcache-vllm.git ../lmcache-vllm

# Step 2: Check vllm installation
output=$(python3 -c "import vllm; print(vllm.__name__)")
if [ "$output" != "vllm" ]; then
    echo -e "\033[31mvLLM is not installed successfully.\033[0m"
    exit 1
fi
echo -e "\033[32mvLLM is installed successfully.\033[0m"

# Step 3: install LMCache and CUDA
cd ../LMCache
pip install -e .
output=$(python3 -c "import torch; import torchac_cuda; print(torchac_cuda.__name__)")
if [ "$output" != "torchac_cuda" ]; then
    echo -e "\033[31mtorchac_cuda is not installed successfully.\033[0m"
    exit 1
fi
output=$(python3 -c "import lmcache; print(lmcache.__name__)")
if [ "$output" != "lmcache" ]; then
    echo -e "\033[31mLMCache is not installed successfully.\033[0m"
    exit 1
fi

# Step 4: install LMCache vllm driver
cd ../lmcache-vllm
pip install -e .
output=$(python3 -c "import lmcache_vllm; print(lmcache_vllm.__name__)" | grep lmcache_vllm)
if [ "$output" != "lmcache_vllm" ]; then
    echo -e "\033[31mLMCache vllm driver is not installed successfully.\033[0m"
    exit 1
fi

# Step 5: Check LMCache server
output=$(python3 -c "import lmcache.server; print(lmcache.server.__name__)" | grep lmcache.server)
if [ "$output" != "lmcache.server" ]; then
    echo -e "\033[31mLMCache server is not installed successfully.\033[0m"
    exit 1
fi

echo -e "\033[32mLMCache is installed successfully.\033[0m"
