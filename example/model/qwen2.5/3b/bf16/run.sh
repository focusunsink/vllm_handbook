# H800
# Processed prompts: 100% 1/1 [00:00<00:00,  2.70it/s, est. speed input: 16187.68 toks/s, output: 140.29 toks/s]
# time used :  0.3717203140258789 avg used 0.007148467577420748

nsys profile \
-o qwen2.5_3b_bf16.nsys-rep  -t cuda,nvtx --delay=5 \
-ftrue \
python3 demo.py ../Qwen2.5-3B/ 6000
