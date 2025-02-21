vllm serve ./deepseek_32b  \
     --tensor-parallel-size 2 \
     --pipeline-parallel-size 2 \
