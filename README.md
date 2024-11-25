# vllm_handbook
How to learn vllm

# 1. install vllm
## flashinfer only support torch2.4.0, vllm 0.6.4 does not support torch 2.4.0, so we choose vllm 0.6.3
pip3 install vllm==0.6.3 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 2. install flashinfer 
## if you don't care performance too much. you don't need this.
pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/


# 3. profiling
# print per token time 
vi usr/local/lib/python3.10/dist-packages/vllm/entrypoints/llm.py:  step_outputs = self.llm_engine.step()
add time capture code above and below.

 
