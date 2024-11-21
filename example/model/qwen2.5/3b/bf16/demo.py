from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch 
import time 
import sys
print('args lens', len(sys.argv))
print('input len', sys.argv[1])

model_path="Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompts = [
    "The future of AI is",
]
test_len=int(sys.argv[1])
prompt_token_ids=tokenizer.encode(prompts[0])
print(len(prompt_token_ids))
prompt_token_ids=prompt_token_ids*(test_len // len(prompt_token_ids))
print(len(prompt_token_ids))

sampling_params = SamplingParams(top_k=1, temperature=1, top_p=1, repetition_penalty=1, max_tokens=52)
llm = LLM(
    model=model_path,
    tensor_parallel_size=1
)

outputs = llm.generate(prompts=None, prompt_token_ids=[prompt_token_ids], sampling_params=sampling_params)
print('warm up done')
for i in range(10):
    outputs = llm.generate(prompts=None, prompt_token_ids=[prompt_token_ids], sampling_params=sampling_params)


torch.cuda.synchronize()
st = time.time()
outputs = llm.generate(prompts=None, prompt_token_ids=[prompt_token_ids], sampling_params=sampling_params)
torch.cuda.synchronize()
ed = time.time()
print("time used : ", ed - st)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
