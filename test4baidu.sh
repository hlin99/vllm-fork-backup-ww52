#!/bin/bash

#set -x
#sleep 3600
#model_path=/mnt/disk002/HF_Models/DeepSeek-R1-Gaudi3/
model_path=/mnt/disk2/hf_models/DeepSeek-R1-G2-static/
#model_path=/mnt/disk2/hf_models/DeepSeek-R1-G2/
ip_addr=10.239.129.24
port=8868
#len_ratio=0.8
max_batch_size=16

export PT_HPU_LAZY_MODE=1

export https_proxy=http://proxy-dmz.intel.com:912
export no_proxy=127.0.0.1
pip install datasets

wait_vllm() {
    #bash start_vllm.sh $model_path $port &>server.log &
    echo "wait vllm server"

    sleep 10
    while true; do
        pids=$(pgrep -f "vllm.entrypoints.openai.api_server")
        if [ -z "$pids" ]; then
            return 1
        fi
        if grep -q ":${port}" server.log; then
            echo "server is ready"
            return 0
        else
            sleep 10
        fi
    done
}



test_lm_eval() {
    start=$(date +%s)
    log_name=benchmark_lm_eval_DeepSeek-R1_cardnumber_8_$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)
    echo "running lm eval testing"
    lm_eval --model local-completions --tasks gsm8k --model_args model=${model_path},base_url=http://${ip_addr}:${port}/v1/completions --batch_size $max_batch_size --log_samples --output_path ./lm_eval_output |& tee ${log_name}.log > /dev/null
        flexable_value=$(grep "flexible-extract" ${log_name}.log | awk -F '|' '{print $8}')

    end=$(date +%s)
    echo "lm_eval flexible-extractvalue: $flexable_value, time taken: $(( end - start )) seconds"
        if (( $(echo "$flexable_value < 0.93" | bc -l) )); then
                echo "The accuracy has issue and testing is stopped"
                exit 1
        fi
}

num_of_p_node = 1

test_benchmark_serving_range() {
    local_input=$1
    local_output=$2
    local_max_concurrency=$3
    local_num_prompts=$(( local_max_concurrency * 5 ))
    local_len_ratio=1
    start=$(date +%s)
	request_rate=$(awk "BEGIN {printf \"%.2f\", 6300 * $num_of_p_node / $local_input}")
	
    echo "running benchmark serving range test, input len: $local_input, output len: $local_output, len ratio: $local_len_ratio, concurrency: $local_max_concurrency, prompt_num: $local_num_prompts"
    log_name=benchmark_serving_DeepSeek-R1_cardnumber_8_datatype_bfloat16_sonnet_batchsize_${local_max_concurrency}_in_${local_input}_out_${local_output}_ratio_${local_len_ratio}_rate_inf_prompts_${local_num_prompts}_$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)

    python3 benchmarks/benchmark_serving.py --backend vllm --model $model_path --trust-remote-code --host $ip_addr --port $port --dataset-name sonnet --dataset-path benchmarks/sonnet.txt --sonnet-input-len $local_input --sonnet-output-len $local_output --sonnet-prefix-len 100 --max_concurrency $local_max_concurrency --num-prompts $local_num_prompts --request-rate $request_rate --seed 0 --ignore_eos --burstiness 1000 --save-result --result-filename ${log_name}.json |& tee ${log_name}.log > /dev/null

    end=$(date +%s)
    output_throughput=$(grep "Output token throughput (tok/s):" ${log_name}.log | awk -F ':' '{print $2}' | xargs)
    mean_tpot=$(grep "Mean TPOT (ms):" ${log_name}.log | awk -F ":" '{print $2}' | xargs)
    echo "Fixed-length dataset, input len: $local_input, output len: $local_output, output throughput (tok/s): $output_throughput, mean TPOT (ms): $mean_tpot, time taken: $(( end - start )) seconds"
}


test_benchmark_serving_range 256 256 672
test_benchmark_serving_range 256 1024 640
test_benchmark_serving_range 1024 256 640
test_benchmark_serving_range 512 512 640
test_benchmark_serving_range 1024 1024 576
test_benchmark_serving_range 2048 1024 512
test_benchmark_serving_range 3584 1536 480
test_benchmark_serving_range 8192 1024 384


test_benchmark_serving_range 256 256 96
test_benchmark_serving_range 256 1024 80
test_benchmark_serving_range 1024 256 80
test_benchmark_serving_range 512 512 80
test_benchmark_serving_range 1024 1024 80
test_benchmark_serving_range 2048 1024 64
test_benchmark_serving_range 3584 1536 48
test_benchmark_serving_range 8192 1024 32
