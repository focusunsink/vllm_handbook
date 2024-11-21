pip3 install vllm==0.6.3 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
# if you need fp8
#git clone https://github.com/neuralmagic/AutoFP8.git
#pip3 install -e AutoFP8/ --force-reinstall torch==2.4.0 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com 

# if you need w8a8
# to prevent this issue https://github.com/vllm-project/llm-compressor/issues/910
pip install compressed-tensors==0.7.1
