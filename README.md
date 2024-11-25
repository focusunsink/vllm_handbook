# vllm_handbook
How to learn vllm

# 1. install 
pip3 install -r requirements.txt


# 2. profiling
# print per token time 
vi usr/local/lib/python3.10/dist-packages/vllm/entrypoints/llm.py:  step_outputs = self.llm_engine.step()
add time capture code above and below.

 
