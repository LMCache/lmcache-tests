git clone git@github.com:LMCache/lmcache-vllm.git
git clone git@github.com:LMCache/lmcache-server.git
git clone git@github.com:LMCache/LMCache.git
git clone git@github.com:LMCache/demo.git
git clone --branch dev/lmcache-integration git@github.com:LMCache/vllm.git
cd vllm
rm pyproject.toml
pip install -r requirements-common.txt
pip install -e .

cd ../LMCache
cd third_party/torchac_cuda
pip install -e .
cd ../..
pip install -e .
cd ../lmcache-vllm
pip install -e .
cd ../lmcache-server
pip install -e .
cd ../demo
pip install openai streamlit
pip install nvtx
#pip install numpy==1.26.4