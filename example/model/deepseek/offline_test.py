from vllm import LLM, SamplingParams


model_path = "./deepseek_14b"
model_path = "./deepseek_32b"

tp = 2
pp = 1
if __name__ == "__main__":

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    llm = LLM(model = model_path, tensor_parallel_size = tp, pipeline_parallel_size = pp)



    for i in range(1000):
        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
