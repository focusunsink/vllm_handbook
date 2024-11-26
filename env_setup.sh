pip3 install -r requirements.txt 
git clone https://github.com/neuralmagic/AutoFP8.git
pip3 install -e AutoFP8/ --force-reinstall torch==2.5.1 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com 
pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4

# if you need w8a8
# to prevent this issue https://github.com/vllm-project/llm-compressor/issues/910
#pip install compressed-tensors==0.7.1

export VLLM_ATTENTION_BACKEND=FLASHINFER
