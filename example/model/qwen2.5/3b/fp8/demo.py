from datasets import load_dataset
from transformers import AutoTokenizer
from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig
import torch
import sys
from vllm import LLM, SamplingParams
import time

import sys



#pretrained_model_dir = "3B"
#quantized_model_dir = "3B_FP8_static"

pretrained_model_dir = sys.argv[1]
quantized_model_dir = sys.argv[2]
input_len = sys.argv[3]


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

text = "Hello hello how are you !"

examples=tokenizer.encode(text)

# Define quantization config with static activation scales
quantize_config = BaseQuantizeConfig(quant_method="fp8", activation_scheme="static")

# Load the model, quantize, and save checkpoint
model = AutoFP8ForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)
model.quantize(torch.LongTensor([examples]).to('cuda'))
model.save_quantized(quantized_model_dir)


print('args lens', len(sys.argv))
print('input len', sys.argv[3])
max_out_tokens = 52


prompts = [
    "The future of AI is",
]
test_len=int(sys.argv[3])
prompt_token_ids=tokenizer.encode(prompts[0])
print(len(prompt_token_ids))
prompt_token_ids=prompt_token_ids*(test_len // len(prompt_token_ids))
