#!/bin/bash
set -e

# Step 1: clone the code
git clone https://github.com/LMCache/lmcache-vllm.git ../lmcache-vllm
git clone https://github.com/LMCache/lmcache-server.git ../lmcache-server
git clone https://github.com/LMCache/LMCache.git ../LMCache
git clone --branch dev/lmcache-integration https://github.com/LMCache/vllm.git ../vllm

# Step 2: install vllm
cd ../vllm
pip install -e .
pip install numpy==1.26.1 # FIXME: this is a temporary fix for the numpy version
output=$(python3 -c "import vllm; print(vllm.__name__)") # Check the installation
if [ "$output" != "vllm" ]; then
    echo -e "\033[31mvLLM is not installed successfully.\033[0m"
    exit 1
fi
echo -e "\033[32mvLLM is installed successfully.\033[0m"

# Step 3: install LMCache and its dependencies
cd ../LMCache
cd third_party/torchac_cuda
pip install -e .
output=$(python3 -c "import torch; import torchac_cuda; print(torchac_cuda.__name__)")
if [ "$output" != "torchac_cuda" ]; then
    echo -e "\033[31mtorchac_cuda is not installed successfully.\033[0m"
    exit 1
fi

cd ../..
pip install -e .
output=$(python3 -c "import lmcache; print(lmcache.__name__)")
if [ "$output" != "lmcache" ]; then
    echo -e "\033[31mLMCache is not installed successfully.\033[0m"
    exit 1
fi

# Step 4: install LMCache vllm driver
cd ../lmcache-vllm
pip install -e .
pip install nvtx openai
output=$(python3 -c "import lmcache_vllm; print(lmcache_vllm.__name__)")
if [ "$output" != "lmcache_vllm" ]; then
    echo -e "\033[31mLMCache vllm driver is not installed successfully.\033[0m"
    exit 1
fi

# Step 5: install LMCache server
cd ../lmcache-server
pip install -e .
output=$(python3 -c "import lmcache_server; print(lmcache_server.__name__)")
if [ "$output" != "lmcache_server" ]; then
    echo -e "\033[31mLMCache server is not installed successfully.\033[0m"
    exit 1
fi
