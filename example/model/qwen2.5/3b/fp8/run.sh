nsys profile \
-o qwen2.5_3b_fp8.nsys-rep -ftrue \
-t cuda,nvtx,osrt --delay=5 \
--cuda-graph-trace=node \
--cudabacktrace=all \
python3 demo.py ../Qwen2.5-3B ./Qwen2.5-3B_fp8 6000
