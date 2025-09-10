python3 benchmarks/benchmark_serving.py --backend vllm --model /mnt/disk2/hf_models/DeepSeek-R1-G2-static/ --dataset-name sonnet --request-rate inf --host 10.239.129.24 --port 8100 --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --trust-remote-code --max-concurrency 512 --num-prompts 128 --ignore-eos --burstiness 1000 --dataset-path benchmarks/sonnet.txt --save-result

