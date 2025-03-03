from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import time
import sys
print('args lens', len(sys.argv))
print("model path: ", sys.argv[1])

input_len = 6000
output_len = 52

model_path=sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompts = [
    "The future of AI is",
]

prompt_token_ids=tokenizer.encode(prompts[0])
print(len(prompt_token_ids))
prompt_token_ids=prompt_token_ids*(input_len // len(prompt_token_ids))
print(len(prompt_token_ids))

sampling_params = SamplingParams(top_k=1, temperature=1, top_p=1, repetition_penalty=1, max_tokens=output_len)
llm = LLM(
    model=model_path,
    tensor_parallel_size=1,
)


print('warm up done')
for i in range(10):
    outputs = llm.generate(prompts=None, prompt_token_ids=[prompt_token_ids], sampling_params=sampling_params)


torch.cuda.synchronize()
st = time.time()
outputs = llm.generate(prompts=None, prompt_token_ids=[prompt_token_ids], sampling_params=sampling_params)
torch.cuda.synchronize()
ed = time.time()
print("time used : ", ed - st)
print("avg decoding time is ", (ed - st) / output_len)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
