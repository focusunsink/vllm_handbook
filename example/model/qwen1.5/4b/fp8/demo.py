from datasets import load_dataset
from transformers import AutoTokenizer
from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig
import torch
import sys
from vllm import LLM, SamplingParams
import time

import sys


model_dir = sys.argv[1]
input_len = sys.argv[2]


tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

text = "Hello hello how are you !"

examples=tokenizer.encode(text)


print('args lens', len(sys.argv))
print('input len', sys.argv[2])
max_out_tokens = 52


prompts = [
    "The future of AI is",
    "what is Google?",
]
test_len=int(sys.argv[2])

prompt_token_ids=tokenizer.encode(prompts[1])
print(len(prompt_token_ids))
prompt_token_ids=prompt_token_ids*(test_len // len(prompt_token_ids))
print(len(prompt_token_ids))

