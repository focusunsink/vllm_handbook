
# 1. if you need fp8
#git clone https://github.com/neuralmagic/AutoFP8.git
#pip3 install -e AutoFP8/ --force-reinstall torch==2.4.0 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com 

# 2. if you need w8a8
# to prevent this issue https://github.com/vllm-project/llm-compressor/issues/910
pip install compressed-tensors==0.7.1

# 3. Flashinfer
# https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/
pip install --no-cache-dir https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.6/flashinfer-0.1.6%2Bcu124torch2.4-cp38-cp38-linux_x86_64.whl
