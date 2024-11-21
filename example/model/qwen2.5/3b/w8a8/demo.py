from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from llmcompressor.transformers import SparseAutoModelForCausalLM

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

quant = True
if quant:
    model = SparseAutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype="auto",
    )
    from datasets import load_dataset

    NUM_CALIBRATION_SAMPLES = 2
    MAX_SEQUENCE_LENGTH = 2048

    # Load and preprocess the dataset
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
    ds = ds.map(tokenize, remove_columns=ds.column_names)

    print("dataset: ", ds)


    from llmcompressor.transformers import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier


    # Configure the quantization algorithms
    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
    ]

    # Apply quantization
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )


    # Save the compressed model
    SAVE_DIR = model_path.split("/")[-1] + "-W8A8-Dynamic-Per-Token"
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)




sampling_params = SamplingParams(top_k=1, temperature=1, top_p=1, repetition_penalty=1, max_tokens=52)
llm = LLM(
    model = SAVE_DIR,
    #model=model_path,
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
    print(f"Prompt: {prompt!r}")
    print(f" Generated text: {generated_text!r}")
    print(f"output token ids is: ", output.outputs[0].token_ids)
